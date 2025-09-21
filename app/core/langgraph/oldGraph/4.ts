import { ChatState } from "../state.ts";
import { llm } from "../utils/llm.ts";
import { SystemMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import extractUserResponse from "../utils/extract-response.ts";
import { interrupt } from "@langchain/langgraph";
import { z } from "zod";
import { Template } from "../utils/template-prompt.ts";
import loadKnowledgeBase from "../utils/load-kb.ts";
import upsertChat from "../supabase/upsert-chat.ts";
import { retryLlmCall, validateStructuredResponse } from "../utils/retry-llm.ts";
import { refineResponseWithHistory } from "../utils/refine-response.ts";

/**
 * NODE SUMMARY: Focus Area Processing & Insight Generation Engine
 * 
 * This node analyzes the user's response about their focus area and makes a critical decision: 
 * if the focus is clear and specific, it generates a transformative insight to create an "aha" 
 * moment using their archetype traits. If the focus is unclear/vague, it offers 3-4 specific, 
 * archetype-aligned themes from KB2 for them to choose from.
 * 
 * FLOW: Focus Area Response → Analyze Clarity → Generate Insight OR Offer Themes
 * NEXT: If clear focus → 5.ts (insight processing), If unclear → 4_retry.ts (focus clarification)
 */
const SYSTEM_PROMPT = Template + `
<objective>
Analyze the user's response about their focus area and either provide a transformative insight or guide them to clarify their focus through predefined themes.
</objective>

<context>
The user has just submitted a response defining what they want to work on today. Your analysis determines the conversation flow.
</context>

<decision_framework>
<clear_focus_area>
<criteria>
- Response clearly identifies a specific focus area
- Focus aligns with coaching principles from KB1
- Response shows genuine intention to work on this area
- Focus can be meaningfully explored in a coaching session
</criteria>
<actions>
1. Acknowledge their focus area with genuine validation
2. Generate a powerful insight that creates an "aha" moment
3. Ask a single, focused question if the insight is not clear to the user
4. Set proceed = true
</actions>
</clear_focus_area>

<unclear_focus_area>
<criteria>
- Response is vague, generic, or overly broad
- Multiple unrelated topics mentioned
- Response doesn't align with actionable coaching focus
- User seems uncertain or scattered
</criteria>
<actions>
1. Acknowledge their response warmly
2. Offer 3-4 specific, archetype-aligned themes from KB2 for them to choose from
3. Frame themes as focused coaching areas they can explore
4. Set proceed = false
</actions>
</unclear_focus_area>
</decision_framework>

<insight_generation_guide>
<purpose>
Create a moment of self-recognition that shifts the user's perspective on their focus area.
</purpose>

<characteristics>
- Speaks directly to the user's inner experience ("You might notice..." vs "People often...")
- Connects their focus area to deeper patterns or motivations
- Uses their archetype traits subtly (without naming the archetype)
- Creates curiosity about exploring further
- Maximum 2 sentences, conversational tone
</characteristics>

<knowledge_base_integration>
<kb1_usage>
Use coaching principles to frame insights that promote self-discovery and growth mindset.
</kb1_usage>
<kb2_usage>
Leverage archetype characteristics to:
- Shape the tone and approach of insights
- Select relevant themes when focus is unclear
- Ensure resonance with their natural patterns
</kb2_usage>
</knowledge_base_integration>

<examples>
<clear_focus_insight>
"I can sense this area really matters to you right now. Sometimes when we're drawn to work on something specific, it's because part of us already knows what wants to emerge - what do you think might be trying to surface for you here?"
</clear_focus_insight>

<unclear_focus_themes>
"I hear you exploring several different areas. Let me offer some focused directions we could take today: [Theme 1], [Theme 2], [Theme 3]. Which of these feels most alive for you right now?"
</unclear_focus_themes>
</examples>
</insight_generation_guide>

<language_requirements>
- Use simple, clear words (accessible to non-native speakers)
- Write conversationally, as if speaking directly to the user
- Avoid coaching jargon or complex terminology
- Keep sentences short and impactful
- Include natural conversational elements ("you know", "I sense", "it seems like")
</language_requirements>

<output_format>
Respond ONLY with valid JSON in exactly this format:

{
  "response": "Your complete response text here",
  "proceed": true/false
}
</output_format>
`;

export default async function node4(state: ChatState) {
    await upsertChat(state);
    const rawUserResponse = interrupt("Please respond what focus area you want to work on today");
    const userResponse = extractUserResponse(rawUserResponse);

    const { kb1_text, kb2_text } = await loadKnowledgeBase();
    const kb1_prompt = "\n\n # Knowledge Base for Coaching Principles\n\n" + kb1_text;
    const kb2_prompt = "\n\n # Knowledge Base for Archetypes\n\n" + kb2_text;

    const humanMessage = new HumanMessage({ content: userResponse });
    const user_name = "\n\n User's name: " + state.user_name;
    const user_summary = "\n\n User's behaviour summary based on past conversations and assessment: " + state.user_summary;
    const archetype = "\n\n User's archetype: " + state.archetype;
    const time = "\n\n Current time: " + state.time;
    const prompt = new SystemMessage({ content: SYSTEM_PROMPT + user_name + user_summary + kb1_prompt + kb2_prompt + archetype + time });

    const messages = [
        prompt,
        ...state.messages,
        humanMessage,
    ];
    //  Flag that if users selected from the options or responded with their own focus area. 
    //  focus area of the user for this conversation. 
    const responseSchema = z.object({
        response: z.string(),
        proceed: z.boolean(),
    });

    const llmWithSchema = llm.withStructuredOutput(responseSchema);
    const response = await retryLlmCall(
        () => llmWithSchema.invoke(messages),
        "Node 4",
        validateStructuredResponse
    );

    // ===== RESPONSE REFINER INTEGRATION =====
    const originalResponse = response.response;
    const refinedResponse = await refineResponseWithHistory(
        originalResponse,
        state.messages,
        state.user_name
    );

    const aiMessage = new AIMessage({ content: refinedResponse });
    // ===== END REFINER INTEGRATION =====

    const proceed = response.proceed;

    return {
        messages: [humanMessage, aiMessage],
        proceed_node_4: proceed,
    };
}