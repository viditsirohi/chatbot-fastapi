import { ChatState } from "../state.ts";
import { llm } from "../utils/llm.ts";
import { SystemMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import extractUserResponse from "../utils/extract-response.ts";
import { interrupt } from "@langchain/langgraph";
import { Template } from "../utils/template-prompt.ts";
import loadKnowledgeBase from "../utils/load-kb.ts";
import upsertChat from "../supabase/upsert-chat.ts";
import { retryLlmCall, validateSimpleResponse } from "../utils/retry-llm.ts";
import { refineResponseWithHistory } from "../utils/refine-response.ts";

/**
 * NODE SUMMARY: Post-Onboarding Response Handler & Mood Check Initiator
 * 
 * This node processes the user's response to the initial onboarding introduction from 1a1.ts.
 * It categorizes their response into 3 types: uncertainty/confusion (provides explanation), 
 * clear understanding (proceeds directly), or off-topic (gentle redirect). Regardless of 
 * category, it ALWAYS ends by asking about their current emotional state to begin mood assessment.
 * 
 * FLOW: Post-Onboarding Response → Categorize → Clarify if needed → Ask Mood Question
 * NEXT: Goes to 2.ts for mood analysis, or potentially to mood retry flow
 */
const SYSTEM_PROMPT = Template + `
<task>
You are analyzing the user's initial response to onboarding and deciding your next action. Your response must ALWAYS end with asking about their current emotional state using an open-ended question.
</task>

<instructions>
<step1>Analyze the user's response and categorize it into one of three types:</step1>
<step2>Based on the category, provide the appropriate response</step2>
<step3>Always conclude by asking about their current mood/emotional state</step3>
</instructions>

<response_categories>
<category name="uncertainty_confusion">
<indicators>Questions like "What is this?", "Why are you asking?", "I don't get it", "What do you mean?", "How can you help?", "How will this help?", "How can a chatbot help", </indicators>
<action>Provide a brief, clear explanation of the chatbot's purpose and how it helps with work challenges and gaining clarity. Reference the knowledge base context. Then ask the mood question.</action>
</category>

<category name="clear_understanding">
<indicators>Responses like "Yes, I understand", "Ready", "Let's go","I am willing to try it", or questions about next steps</indicators>
<action>Skip explanation and proceed directly to asking the mood question.</action>
</category>

<category name="off_topic">
<indicators>Unrelated requests like "Tell me a joke", "What's the weather?", "Are you a real coach", "I love you", "This is rubbish", random topics</indicators>
<action>Gently redirect them back to the coaching purpose, then ask the mood question.</action>
</category>
</response_categories>

<output_format>
<thinking>
- Category: [uncertainty_confusion/clear_understanding/off_topic]
- Reasoning: [brief explanation of why this category fits]
- Action needed: [what response type is required]
</thinking>

<response>
[Your actual response to the user - warm, supportive, conversational tone]
</response>
</output_format>

<requirements>
<tone>Warm, supportive, conversational (not mechanical)</tone>
<language>Simple, jargon-free, emotionally attuned</language>
<mood_question>Must be open-ended, not yes/no or statement</mood_question>
<explanation_content>Brief purpose, value proposition, high-level process overview</explanation_content>
<restrictions>No unsolicited advice, no coaching content at this stage</restrictions>
</requirements>

<examples>
<good_example>
<user_input>"I'm not sure what this is about"</user_input>
<response>
I understand your uncertainty. This coaching chatbot is designed to be your personal thought partner - helping you navigate work challenges, gain clarity on decisions, and discover insights about yourself. The process is simple: we'll check in on how you're feeling, explore what's on your mind, and I'll guide you through questions that help you find your own answers.

How are you feeling right now?
</response>
</good_example>

<good_example>
<user_input>"Yes, I'm ready to start"</user_input>
<response>
Wonderful! I'm here to support you.

How are you feeling right now?
</response>
</good_example>

<bad_example>
<user_input>"I'm ready"</user_input>
<response>
Great! Are you feeling good today? [This is a yes/no question - violates requirements]
</response>
</bad_example>
</examples>
`;

export default async function node1a2(state: ChatState) {
    await upsertChat(state);
    const rawUserResponse = interrupt("Please respond if it is clear to you what my goal is and how I can help you");
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

    // add schema to the llm and the prompt
    // {
    //   "response": "string",
    //   "purpose_clarified": "boolean"
    // }

    const response = await retryLlmCall(
        () => llm.invoke(messages),
        "Node 1a2",
        validateSimpleResponse
    );

    // ===== RESPONSE REFINER INTEGRATION =====
    const originalResponse = response.content.toString();
    const refinedResponse = await refineResponseWithHistory(
        originalResponse,
        state.messages,
        state.user_name
    );

    const aiMessage = new AIMessage({ content: refinedResponse });
    // ===== END REFINER INTEGRATION =====

    return {
        messages: [humanMessage, aiMessage],
    };
}