import { ChatState } from "../state.ts";
import { llm } from "../utils/llm.ts";
import { SystemMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import extractUserResponse from "../utils/extract-response.ts";
import { interrupt } from "@langchain/langgraph";
import { z } from "zod";
import { Template } from "../utils/template-prompt.ts";
import loadKnowledgeBase from "../utils/load-kb.ts";
import upsertChat from "../supabase/upsert-chat.ts";
import upsertMood from "../supabase/upsert-mood.ts";
import { retryLlmCall, validateStructuredResponse } from "../utils/retry-llm.ts";
import { refineResponseWithHistory } from "../utils/refine-response.ts";


/**
 * NODE SUMMARY: Mood Check-In Response Analyzer & Router
 * 
 * This node analyzes the user's response to mood/emotional state questions and categorizes it 
 * into three types: sufficient clarity (clear emotions like "tired", "overwhelmed"), unclear/vague 
 * (responses like "okay", "fine"), or irrelevant/off-topic. Based on the category, it either 
 * proceeds to mood exploration options (3.ts) or routes to retry for clearer mood expression (2_retry.ts).
 * 
 * FLOW: Mood Response → Analyze Clarity → Route (Proceed vs Retry)
 * NEXT: If clear → 3.ts (mood exploration choice), If unclear → 2_retry.ts (mood clarification)
 */
const SYSTEM_PROMPT = Template + `
<task>
Analyze the user's mood check-in response and categorize it to determine next steps. Your response must include acknowledgement, appropriate empathy, and either proceed or help them explore further. Always respond in a natural, conversational tone like a real human coach. IMPORTANT: When the user's mood is clearly identified, extract the specific mood word for logging to the backend.
</task>

<context>
The user has just responded to a mood/emotional state question. You must evaluate their response quality and provide appropriate coaching support.
</context>

<analysis_framework>
<step1>Categorize the user's response into one of three types</step1>
<step2>Provide appropriate acknowledgement and empathy</step2>
<step3>Take the prescribed action for that category</step3>
<step4>Set the correct proceed value</step4>
<step5>Extract the specific mood word when clearly identified for backend logging</step5>
</analysis_framework>

<response_categories>
<category name="sufficient_mood_clarity">
<indicators>Clear emotional descriptors like "tired", "overwhelmed", "excited", "anxious", "great", "stressed", "calm", specific mood words</indicators>
<actions>
- Validate their sharing with brief acknowledgment
- Provide empathetic response (especially for negative emotions)
- Ask if they want to explore this mood further or move to different focus
- Extract the specific mood word for the mood field
- Set proceed = true
</actions>
</category>

<category name="unclear_or_vague">
<indicators>Responses like "okay", "fine", "I don't know", "mixed", "hard to say", "not sure how to describe it", vague or uncertain language</indicators>
<actions>
- Acknowledge their attempt to share
- Validate that understanding mood can be challenging
- Use a conversational filler phrase + one of the 4 metaphorical questions (selected randomly)
- Keep tone conversational and natural
- Set proceed = false
</actions>
</category>

<category name="irrelevant_or_off_topic">
<indicators>Questions about the process, random text, off-topic responses, technical questions, unrelated content</indicators>
<actions>
- Provide brief context about mood check-in purpose
- Gently redirect back to mood question
- Set proceed = false
</actions>
</category>
</response_categories>

<output_format>
<thinking>
- Category: [sufficient_mood_clarity/unclear_or_vague/irrelevant_or_off_topic]
- Reasoning: [why this category fits the response]
- Empathy needed: [yes/no and why]
- Action required: [what type of response to give]
</thinking>

<response>
[Your actual response - warm, validating, with appropriate next step]
</response>

<proceed_value>
[true/false based on category]
</proceed_value>

<mood_value>
[specific emotion word like "anxious", "excited", "overwhelmed" when clearly identified, leave empty for unclear responses]
</mood_value>
</output_format>

<response_guidelines>
<acknowledgement>Always acknowledge their sharing before proceeding</acknowledgement>
<empathy>Provide extra support for negative emotions without being overly sympathetic</empathy>
<questions>When asking exploration questions, use a conversational filler phrase like "Let's make this simpler" or "Let's make this easier" before the metaphorical question</questions>
<metaphorical_questions>Use ONLY the 5 specified metaphorical questions, selecting any one randomly each time</metaphorical_questions>
<tone>Warm, supportive, conversational coaching tone - sound like a real human coach, not robotic</tone>
<brevity>Keep responses concise but complete and natural</brevity>
</response_guidelines>

<experiential_questions_bank>
<metaphorical_questions_only>
Use ONLY these 5 questions, select ONE randomly each time in any order of the questions. Always precede with a conversational filler phrase:

Filler phrases to use randomly:
- "Let's make this simpler..."
- "Let's make this easier..."
- "Let me try a different approach..."
- "Here's another way to think about it..."

The 5 metaphorical questions (choose one randomly):
1. "If your current mood was a color, what color would it be?"
2. "If your current mood was weather, what would the forecast be?"
3. "If your current mood was a texture, what would it feel like?"
4. "If your current mood had a temperature, what temperature would it be?"
5. Can you pick an emoji that is closest to how you are feeling right now?
</metaphorical_questions_only>
</experiential_questions_bank>

<examples>
<good_example>
<user_input>"I'm feeling really overwhelmed with work"</user_input>
<thinking>Category: sufficient_mood_clarity, clear emotion, empathy needed for negative state. Mood extraction: "overwhelmed"</thinking>
<response>Thank you for sharing that with me. Feeling overwhelmed at work can be really draining. Would you like to explore what's contributing to that overwhelmed feeling, or would you prefer to focus on a different area today?</response>
<proceed>true</proceed>
<mood>overwhelmed</mood>
</good_example>

<good_example>
<user_input>"I don't really know, just okay I guess"</user_input>
<thinking>Category: unclear_or_vague, needs exploration, use filler phrase + metaphorical question. Mood extraction: unclear, leave empty</thinking>
<response>I appreciate you taking a moment to check in with yourself. Sometimes it can be hard to pinpoint exactly how we're feeling. Let's make this easier - if your current mood was weather, what would the forecast be?</response>
<proceed>false</proceed>
<mood></mood>
</good_example>

<good_example>
<user_input>"What is this question for?"</user_input>
<thinking>Category: irrelevant_or_off_topic, needs redirection. Mood extraction: unclear, leave empty</thinking>
<response>This mood check-in helps me understand where you're at emotionally so I can better support you in our conversation. How are you feeling right now?</response>
<proceed>false</proceed>
<mood></mood>
</good_example>
</examples>

<restrictions>
<no_repetition>Don't simply repeat what the user said back to them</no_repetition>
<no_advice>Don't offer solutions or advice at this stage</no_advice>
<no_assumptions>Don't assume reasons for their mood</no_assumptions>
<one_question>Ask only one exploration question if needed</one_question>
</restrictions>
`;



export default async function node2(state: ChatState) {
    await upsertChat(state);
    const rawUserResponse = interrupt("Please respond to how you're feeling today");
    const userResponse = extractUserResponse(rawUserResponse);

    const { kb1_text } = await loadKnowledgeBase();
    const kb1_prompt = "\n\n # Knowledge Base for Coaching Principles\n\n" + kb1_text;

    // IMPORTANT: Extract mood when clear for backend logging
    // Only return specific mood word when clearly identified, not for vague responses
    const responseSchema = z.object({
        response: z.string(),
        proceed: z.boolean(),
        mood: z.string().optional(),
    });

    const humanMessage = new HumanMessage({ content: userResponse });
    const user_name = "\n\n User's name: " + state.user_name;
    const user_summary = "\n\n User's behaviour summary based on past conversations and assessment: " + state.user_summary;
    const archetype = "\n\n User's archetype: " + state.archetype;
    const time = "\n\n Current time: " + state.time;
    const prompt = new SystemMessage({ content: SYSTEM_PROMPT + user_name + user_summary + kb1_prompt + archetype + time });
    const messages = [
        prompt,
        ...state.messages,
        humanMessage,
    ];

    const llmWithSchema = llm.withStructuredOutput(responseSchema);
    const response = await retryLlmCall(
        () => llmWithSchema.invoke(messages),
        "Node 2",
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
    const mood = response.mood;

    // Store mood in state and log to backend when determined
    const stateUpdate: any = {
        messages: [humanMessage, aiMessage],
        proceed_node_2: proceed,
    };

    if (mood && mood !== "" && mood !== "assessing") {
        stateUpdate.mood = mood;
        // Log mood to backend asynchronously
        const updatedState = { ...state, ...stateUpdate };
        await upsertMood(updatedState);
    }

    return stateUpdate;
}