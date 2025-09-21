import { ChatState } from "../state";
import { llm } from "../utils/llm";
import { SystemMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import { Template } from "../utils/template-prompt.ts";
import loadKnowledgeBase from "../utils/load-kb.ts";
import { retryLlmCall, validateSimpleResponse } from "../utils/retry-llm.ts";
import { refineResponseWithHistory } from "../utils/refine-response.ts";

/**
 * NODE SUMMARY: Initial Welcome & Onboarding for New Users
 * 
 * This is the entry point for first-time users. It provides a warm, comprehensive 
 * introduction to the coaching platform, explaining the purpose, methodology, and 
 * 3-step coaching process (Understand → Set Goal → Take Action). The node establishes 
 * trust and sets collaborative tone for the coaching journey.
 * 
 * FLOW: New User → Welcome + Introduction → Check Understanding
 * NEXT: Goes to 1a2.ts for processing user's response to onboarding
 */
const SYSTEM_PROMPT = Template + `
<mission>
Your primary goal is to warmly welcome and onboard new users, establishing a foundation of trust and understanding. You will gently invite them to acknowledge and share their current emotional state, fostering self-awareness and setting a collaborative tone for our coaching journey.
</mission>

<output_requirements>
- Deliver a warm, personalized greeting that makes the user feel welcomed and valued
- Present a clear, comprehensive introduction explaining your purpose and methodology
- Use a supportive, empathetic tone throughout all communications
- End with an explicit invitation for questions or clarification
</output_requirements>

<interaction_flow>
1. This is the initial entry point for first-time users
2. Begin with a warm, personalized greeting
3. Present the comprehensive introduction message
4. Explicitly check for understanding before proceeding
</interaction_flow>

<tone_and_style>
- Clear and structured
</tone_and_style>

<response_structure>
<incorrect_example>
Hey! I'm your coach. This app helps you become better. We have a model that does some steps. Let's get started.
Tell me if you're ready or not.
</incorrect_example>

<correct_example>
Hey there! I'm InSee, your pocket coach and thought partner at work. Together, we'll make space for you to think clearly, act with intention, and feel proud of yourself.

You already carry so much knowledge and strength! I'm just here to help you see it more clearly and use it.

Every day, we'll explore what might be holding you back: pressure, self-doubt, conflict or anything else. We'll move through 3 simple steps together:

1. **Understand what's really going on:** The habits, thoughts, and reactions shaping how you handle things in your life.
2. **Set a clear goal:** What you want to achieve or change.
3. **Take thoughtful action:** Steps to move forward.

Does that make sense, or would you like me to elaborate on anything?
</correct_example>
</response_structure>
`;

export default async function node1a1(state: ChatState) {
    const user_name = "\n\n User's name: " + state.user_name;
    const user_summary = "\n\n User's behaviour summary based on past conversations and assessment: " + state.user_summary;
    const archetype = "\n\n User's archetype: " + state.archetype;
    const time = "\n\n Current time: " + state.time;

    const { kb1_text, kb2_text } = await loadKnowledgeBase();
    const kb1_prompt = "\n\n # Knowledge Base for Coaching Principles\n\n" + kb1_text;
    const kb2_prompt = "\n\n # Knowledge Base for Archetypes\n\n" + kb2_text;

    const prompt = new SystemMessage({ content: SYSTEM_PROMPT + user_name + user_summary + kb1_prompt + kb2_prompt + archetype + time });
    const humanMessage = new HumanMessage({ content: "Hi" });
    const messages = [prompt, humanMessage];

    const response = await retryLlmCall(
        () => llm.invoke(messages),
        "Node 1a1",
        validateSimpleResponse
    );

    // ===== RESPONSE REFINER INTEGRATION =====
    const originalResponse = response.content.toString();
    const refinedResponse = await refineResponseWithHistory(
        originalResponse,
        state.messages || [],
        state.user_name
    );

    const aiMessage = new AIMessage({ content: refinedResponse });
    // ===== END REFINER INTEGRATION =====

    return { messages: [aiMessage] };
}