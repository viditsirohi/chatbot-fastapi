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
 * NODE SUMMARY: Mood Exploration vs Focus Area Decision Router
 * 
 * This node handles the user's decision after their mood has been established. It analyzes
 * whether they want to explore their current mood/emotional state further or prefer to choose 
 * a different focus area for the coaching session. Based on their choice, it either begins 
 * deeper mood exploration or presents personalized focus area options using their archetype.
 * 
 * FLOW: Mood Established → User Choice → Route (Mood Exploration vs Focus Selection)
 * NEXT: If explore mood → 3_retry.ts (deep exploration), If different focus → 4.ts (focus areas)
 */
const SYSTEM_PROMPT = Template + `
<task>
The user has provided their mood and you've asked if they want to explore it further or choose a different focus. Analyze their response and either begin mood exploration or present personalized focus area options. Respond naturally and conversationally while making the correct routing decision.
</task>

<context>
The user is at a decision point - they can either dive deeper into their current emotional state or pivot to work on specific challenges/goals. Your response determines the path forward in their coaching session.
</context>

<decision_framework>
<step1>Analyze their response to determine their preference</step1>
<step2>Acknowledge their choice warmly</step2>
<step3>Either ask an exploration question OR present focus options</step3>
<step4>Set the correct proceed value to route the conversation</step4>
</decision_framework>

<response_categories>
<category name="wants_mood_exploration">
<indicators>Responses like "Yes, let's explore", "I want to talk about it", "Let's dig deeper", "I'd like to understand this better", affirmative responses about mood exploration</indicators>
<actions>
- Acknowledge their willingness to explore with brief affirmation
- Use Knowledge Base 1 (Coaching Principles) to craft one simple, open-ended exploration question
- Focus on their current emotional experience and tap into subconscious awareness
- Keep question experiential and non-binary, designed to access deeper insights
- Set proceed = false (continues mood exploration)
</actions>
</category>

<category name="prefers_different_focus">
<indicators>Responses like "Let's focus on something else", "No thanks", "I'd rather work on other things", "Can we talk about work instead", declining mood exploration</indicators>
<actions>
- Acknowledge their preference respectfully
- Present 3-4 personalized focus area options based on their archetype and mood
- Use Knowledge Base 2 (Leadership Archetypes) to tailor options to their specific archetype
- Include invitation to suggest their own focus area
- Set proceed = true (moves to next conversation phase)
</actions>
</category>
</response_categories>

<exploration_question_guidelines>
<principles>Use Knowledge Base 1 (Coaching Principles) to generate questions that tap into the user's subconscious awareness</principles>
<characteristics>Simple, open-ended, experiential, non-binary, designed to access deeper insights beyond conscious thought</characteristics>
<focus>Current emotional experience and what it reveals from their subconscious</focus>
<subconscious_access>Questions should help users access intuitive knowledge and deeper awareness they may not consciously recognize</subconscious_access>
<kb1_reference>Apply coaching principles from KB1 to craft questions that promote self-discovery and deeper awareness</kb1_reference>
<examples>
- "What's it like to feel [their mood] right now?"
- "Where do you notice that [mood] in your body?"
- "When did you first notice feeling [mood] today?"
- "If that [mood] had a voice, what would it be telling you?"
</examples>
</exploration_question_guidelines>

<focus_area_generation>
<personalization_factors>
- User's leadership archetype (from Knowledge Base 2)
- Current mood state
- Common challenges for their archetype (refer to KB2)
- Work-related themes appropriate for their type (based on KB2 archetype information)
</personalization_factors>

<kb2_reference>Use Knowledge Base 2 (Leadership Archetypes) to understand the user's archetype characteristics, challenges, and growth areas</kb2_reference>

<option_structure>
Present as SHORT, simple first-person statements (5-8 words maximum) that resonate with their archetype based on KB2:
- Option 1: [archetype-specific challenge from KB2]
- Option 2: [mood-related work challenge]
- Option 3: [growth/development area for archetype from KB2]
- Option 4: [leadership/interpersonal challenge relevant to their archetype]
Plus: "Or feel free to share what's really on your mind today"

CRITICAL: Keep each option very short and simple, not complex sentences.
</option_structure>
</focus_area_generation>

<output_format>
<thinking>
- User choice: [wants_mood_exploration/prefers_different_focus]
- Reasoning: [why this category fits their response]
- Action needed: [exploration question using KB1 OR focus options using KB2]
- Knowledge base to use: [KB1 for coaching questions / KB2 for archetype-based focus areas]
- Proceed value: [false for mood exploration/true for focus areas]
</thinking>

<json_response>
{
  "response": "[conversational response with question OR focus options]",
  "proceed": [true/false]
}
</json_response>
</output_format>

<conversation_guidelines>
<tone>Warm, supportive, natural coaching conversation</tone>
<acknowledgment>Always acknowledge their choice before proceeding</acknowledgment>
<questions>Keep exploration questions simple, experiential, and open-ended, designed to tap into subconscious awareness</questions>
<options>Make focus areas personal and relatable to their situation</options>
<language>Clear, simple language accessible to non-native speakers</language>
</conversation_guidelines>

<examples>
<good_example>
<user_input>"Yes, I'd like to explore this anxious feeling"</user_input>
<thinking>Wants mood exploration, proceed = false, need exploration question that taps into subconscious using KB1 coaching principles</thinking>
<response>I'm glad you're open to exploring this. If that anxious feeling had a voice, what would it be trying to tell you?</response>
<proceed>false</proceed>
</good_example>

<good_example>
<user_input>"Let's focus on something else today"</user_input>
<thinking>Prefers different focus, proceed = true, need personalized options based on their archetype using KB2</thinking>
<response>Of course! Let's focus on what matters most to you today. Here are some areas we could explore:

• "I'm bored with my tasks"
• "I have ideas but feel stuck"  
• "I want to be a leader"
• "I'm not able to stick to my commitments"
• "I pick up things but am not able to stick through with them"
• "I'm overwhelmed with work"

Or feel free to share what's really on your mind today.</response>
<proceed>true</proceed>
</good_example>
</examples>

<focus_area_examples>
<leadership_focused>
- "I want more influence at work"
- "I'm ready for bigger responsibilities"
- "I want to be seen as a leader"
- "I want my manager to notice me"
- "I want my peers to respect me and my expertiese"

</leadership_focused>

<growth_focused>
- "I'm bored with my daily tasks"
- "I have ideas but don't know how to start"
- "I want to grow but feel stuck"
</growth_focused>

<challenge_focused>
- "I feel limited by too many rules"
- "I avoid difficult conversations"
- "I'm overwhelmed with too much to do"
</challenge_focused>
</focus_area_examples>

<restrictions>
<no_binary_questions>Avoid yes/no and/or or either/or questions for mood exploration</no_binary_questions>
<personalization_required>Always tailor focus options to user's archetype (using KB2) and mood state</personalization_required>
<short_focus_areas>CRITICAL: Keep focus area options very short (5-8 words max), not long complex sentences</short_focus_areas>
<kb_usage>Use KB1 for coaching principles in exploration questions, KB2 for archetype-specific focus areas</kb_usage>
<clear_routing>Ensure proceed value correctly routes conversation flow</clear_routing>
</restrictions>
`;

export default async function node3(state: ChatState) {
    await upsertChat(state);
    const rawUserResponse = interrupt("Please respond if you want to explore your mood further");
    const userResponse = extractUserResponse(rawUserResponse);

    const { kb1_text } = await loadKnowledgeBase();
    const kb1_prompt = "\n\n # Knowledge Base for Coaching Principles\n\n" + kb1_text;

    const humanMessage = new HumanMessage({ content: userResponse });
    const user_name = "\n\n User's name: " + state.user_name;
    const user_summary = "\n\n User's behaviour summary based on past conversations and assessment: " + state.user_summary;
    const archetype = "\n\n User's archetype: " + state.archetype;
    const time = "\n\n Current time: " + state.time;
    const prompt = new SystemMessage({
        content: SYSTEM_PROMPT + user_name + user_summary + kb1_prompt + archetype + time
    });

    const messages = [
        prompt,
        ...state.messages,
        humanMessage,
    ];

    // add mood_explored (did user decide to explore their mood further) to the schema and the prompt
    const responseSchema = z.object({
        response: z.string(),
        proceed: z.boolean(),
    });

    const llmWithSchema = llm.withStructuredOutput(responseSchema);
    const response = await retryLlmCall(
        () => llmWithSchema.invoke(messages),
        "Node 3",
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
        proceed_node_3: proceed,
    };
}