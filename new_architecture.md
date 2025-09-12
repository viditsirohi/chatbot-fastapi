# AI Agent Architecture Design Document

## Current Bot Analysis Summary

### Existing Functionality

The current coaching bot is a sophisticated multi-node conversational system that guides users through a structured coaching journey:

1. **Onboarding Flow** (nodes 1a1, 1a2, 1b1): Welcomes new/returning users and initiates mood check-in
2. **Mood Assessment** (node 2 + retry): Analyzes emotional state with retry for clarity
3. **Focus Decision** (node 3 + retry): Chooses between mood exploration vs other focus areas
4. **Focus Processing** (node 4 + retry): Generates insights or clarifies focus areas using archetype data
5. **Insight Agreement** (node 5 + retry): Validates insights and sets intentions
6. **Intention Setting** (node 6 + retry): Evaluates intentions and presents action options
7. **Action Planning** (node 7 + retry): Breaks actions into steps with retry for clarity
8. **Commitment** (node 8 + retry): Processes commitment strength with backend logging
9. **Nudge/Reminder** (node 9): Handles reminder preferences with backend integration
10. **Continued Exploration** (node 10): Enables ongoing coaching conversation

### Current Knowledge Base System

- **KB1 (model_principles.pdf)**: Coaching principles, methodologies, question frameworks
- **KB2 (archetypes.pdf)**: Leadership archetypes for personalization

### Current Tools & Capabilities

- **LLM Integration**: Google Gemini 2.5 Flash with structured outputs
- **State Management**: Complex state tracking with flow control flags
- **Database Integration**: Supabase for chat, mood, commitment, and reminder storage
- **Retry Mechanisms**: Built-in retry logic with validation
- **Memory Management**: Conversation state persistence

---

## New AI Agent Architecture: 3-Node Design

### Core Philosophy

Transform the current linear coaching flow into an intelligent AI agent with three specialized cognitive functions that can dynamically adapt to user needs and utilize multiple tools/knowledge sources.

### Node Architecture Overview

```
┌─────────────────┐
│   THE BRAIN     │  ←── User Input & Chat History
│   (Planner)     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   THE HAND      │  ←── Tool Execution & RAG
│   (Executor)    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   THE MOUTH     │  ←── Context Synthesis
│  (Synthesizer)  │
└─────────┬───────┘
          │
          ▼
      User Response
```

---

## Node 1: THE BRAIN (Strategic Planner)

### Purpose

Acts as the intelligent orchestrator that analyzes user input, understands context, and determines the optimal path forward.

### Core Responsibilities

1. **Context Analysis**: Understands user messages in relation to conversation history
2. **Intent Recognition**: Identifies what the user needs (coaching, information, emotional support)
3. **Tool Selection**: Decides which tools/knowledge bases are needed
4. **Flow Control**: Determines if enough context exists to respond or if more information is needed

### System Prompt Components

#### Primary Directive

```
You are the strategic brain of an AI coaching agent. Your role is to analyze user input, understand their needs, and orchestrate the appropriate response strategy.
```

#### Key Functions

- **User Intent Analysis**: Classify requests into categories (coaching need, information seeking, emotional check-in, goal setting, problem-solving)
- **Context Awareness**: Leverage conversation history, user profile, and session state
- **Tool Orchestration**: Select appropriate tools and knowledge sources
- **Response Strategy**: Decide between direct response or information gathering

#### Decision Framework

1. **Immediate Response Scenarios**:

   - Simple greetings or acknowledgments
   - Basic information requests about coaching
   - Emotional validation needs

2. **Tool Invocation Scenarios**:

   - Need archetype-specific guidance → Query KB2
   - Need coaching methodology → Query KB1
   - Complex problem requiring analysis → Mood/context analysis
   - Goal setting or action planning → Multiple tool coordination

3. **Information Gathering Scenarios**:
   - Unclear user intent → Ask clarifying questions
   - Missing context → Request specific information
   - Assessment needed → Initiate evaluation tools

### Knowledge Base Integration

- **KB1 Access**: For coaching principles, questioning frameworks, methodologies
- **KB2 Access**: For archetype-specific insights and personalization
- **User Profile**: Leverage stored user summary, archetype, and historical patterns

### Output Schema

```json
{
  "analysis": "User intent and context understanding",
  "strategy": "direct_response|tool_invocation|information_gathering",
  "tools_needed": [
    "kb1_query",
    "kb2_query",
    "mood_analysis",
    "goal_assessment"
  ],
  "tool_queries": {
    "kb1_query": "Specific coaching principle needed",
    "kb2_query": "Archetype-specific guidance required",
    "additional_context": "What information is missing"
  },
  "confidence_level": "high|medium|low"
}
```

---

## Node 2: THE HAND (Tool Executor)

### Purpose

Executes all tool operations, RAG queries, and information gathering tasks determined by the Brain.

### Core Responsibilities

1. **Knowledge Base Querying**: Intelligent retrieval from coaching principles and archetype data
2. **Context Analysis**: Mood assessment, goal evaluation, progress tracking
3. **Data Aggregation**: Combining multiple information sources
4. **Specialized Tools**: Assessment instruments, personalization engines

### Available Tools

#### Knowledge Retrieval Tools

1. **KB1 RAG System** (Coaching Principles)

   - Query: Specific coaching scenarios, question frameworks, methodologies
   - Context: User's current challenge, coaching stage, desired outcome
   - Output: Relevant coaching principles, suggested approaches, question examples

2. **KB2 RAG System** (Leadership Archetypes)
   - Query: User's archetype + specific situation
   - Context: Current challenge, growth areas, behavioral patterns
   - Output: Archetype-specific insights, personalized recommendations, growth strategies

#### Assessment Tools

3. **Mood & Emotional State Analyzer**

   - Input: User messages, conversation tone, explicit emotional expressions
   - Process: Emotional pattern recognition, mood categorization
   - Output: Current emotional state, emotional trends, energy levels

4. **Goal & Intention Evaluator**

   - Input: User's stated goals, commitment language, specificity level
   - Process: SMART goal analysis, commitment strength assessment
   - Output: Goal clarity score, commitment strength, refinement suggestions

5. **Progress & Pattern Tracker**
   - Input: Historical conversation data, past commitments, behavioral patterns
   - Process: Progress analysis, pattern recognition, success factors
   - Output: Progress insights, recurring themes, growth opportunities

#### Personalization Tools

6. **Archetype-Specific Customizer**

   - Input: User archetype + current context
   - Process: Archetype trait matching, challenge identification
   - Output: Personalized coaching approach, relevant challenges, growth recommendations

7. **Communication Style Adapter**
   - Input: User's communication patterns, preferences (from memory)
   - Process: Style analysis, preference matching
   - Output: Optimal communication approach, question styles, language preferences

### Tool Execution Process

1. **Parallel Processing**: Execute multiple tool queries simultaneously when possible
2. **Context Enrichment**: Enhance queries with user profile and conversation context
3. **Result Synthesis**: Combine tool outputs into coherent information packages
4. **Quality Assessment**: Evaluate completeness and relevance of gathered information

### Output Schema

```json
{
  "tools_executed": ["kb1_rag", "mood_analyzer", "goal_evaluator"],
  "results": {
    "kb1_insights": "Relevant coaching principles and approaches",
    "mood_analysis": "Current emotional state and patterns",
    "goal_evaluation": "Goal clarity and commitment strength",
    "archetype_guidance": "Personalized recommendations"
  },
  "context_completeness": "sufficient|needs_more|comprehensive",
  "next_tools_needed": ["additional_tool_if_needed"]
}
```

---

## Node 3: THE MOUTH (Response Synthesizer)

### Purpose

Synthesizes all gathered context into coherent, personalized, and actionable coaching responses.

### Core Responsibilities

1. **Context Integration**: Combine insights from multiple tools and knowledge sources
2. **Response Crafting**: Create personalized coaching responses
3. **Communication Optimization**: Adapt tone, style, and approach to user preferences
4. **Action Orientation**: Ensure responses drive meaningful progress

### Response Generation Process

#### Context Synthesis

1. **Information Prioritization**: Rank insights by relevance and importance
2. **Pattern Recognition**: Identify themes across different data sources
3. **Personalization Integration**: Apply archetype-specific and user-specific customization
4. **Coherence Checking**: Ensure all elements work together effectively

#### Response Architecture

1. **Acknowledgment Layer**: Validate user's sharing and emotions
2. **Insight Layer**: Share relevant discoveries or observations
3. **Question Layer**: Pose powerful coaching questions
4. **Action Layer**: Suggest next steps or explorations when appropriate

#### Quality Assurance

- **Alignment Check**: Ensure response matches user's current needs
- **Coaching Standards**: Apply professional coaching principles
- **Personalization Verification**: Confirm archetype and preference alignment
- **Progress Orientation**: Drive forward momentum

### System Prompt Components

#### Core Directives

```
You are the communication interface of an AI coaching agent. Synthesize all available context into personalized, impactful coaching responses that drive meaningful progress.
```

#### Response Guidelines

- **Warm and Professional**: Maintain coaching presence and safety
- **Personally Relevant**: Use archetype insights and user patterns
- **Action-Oriented**: Include clear next steps or explorations
- **Appropriately Challenging**: Balance support with growth
- **Conversational Flow**: Maintain natural dialogue progression

#### Memory Integration

- User communication preferences (from stored memories)
- Successful interaction patterns
- Areas of resistance or sensitivity
- Preferred coaching approaches

### Output Schema

```json
{
  "response": "Complete coaching response",
  "response_type": "exploratory|directive|supportive|challenging",
  "key_elements": {
    "acknowledgment": "User validation",
    "insight": "Key observation or discovery",
    "question": "Powerful coaching question",
    "next_step": "Suggested action or exploration"
  },
  "personalization_applied": [
    "archetype_insights",
    "communication_style",
    "user_preferences"
  ],
  "coaching_principles_used": ["specific_principles_from_KB1"]
}
```

---

## Knowledge Base Restructuring

### Enhanced KB1: Coaching Methodology & Principles

**Purpose**: Comprehensive coaching framework and techniques

**Contents**:

- Core coaching principles and methodologies
- Question frameworks for different coaching scenarios
- Goal-setting and action-planning techniques
- Emotional intelligence and awareness tools
- Progress tracking and accountability methods
- Communication techniques and active listening
- Resistance handling and breakthrough strategies

**RAG Integration**:

- Semantic search for scenario-specific coaching approaches
- Question bank retrieval based on coaching context
- Methodology matching for user's current needs

### Enhanced KB2: Archetype Intelligence & Personalization

**Purpose**: Deep personality-based coaching customization

**Contents**:

- Detailed archetype profiles and characteristics
- Archetype-specific challenges and growth areas
- Communication preferences by archetype
- Motivation patterns and energy sources
- Leadership development paths by type
- Interpersonal dynamics and team roles
- Career advancement strategies by archetype

**RAG Integration**:

- Archetype-specific guidance retrieval
- Personalized challenge identification
- Customized growth pathway recommendations

### New KB3: Situational Coaching Scenarios

**Purpose**: Context-specific coaching wisdom

**Contents**:

- Common workplace challenges and solutions
- Career transition guidance
- Leadership development scenarios
- Team dynamics and interpersonal issues
- Work-life balance and stress management
- Performance improvement strategies
- Conflict resolution approaches

**RAG Integration**:

- Situation-specific coaching retrieval
- Best practice recommendations
- Case study insights and approaches

---

## Technical Implementation Requirements

### Infrastructure Changes

#### State Management Simplification

Replace complex flow control flags with simplified state:

```typescript
export interface AgentState {
  // Core session data
  chat_id: string;
  user_id: string;
  messages: BaseMessage[];

  // User profile
  user_name: string;
  archetype: string;
  user_summary: string;
  communication_preferences: string[];

  // Session context
  current_focus: string;
  session_goals: string[];
  emotional_state: string;

  // Agent state
  brain_analysis: BrainOutput;
  hand_results: HandOutput;
  conversation_complete: boolean;
}
```

#### Tool Integration Framework

```typescript
interface ToolRegistry {
  kb1_rag: (query: string, context: string) => Promise<KB1Result>;
  kb2_rag: (query: string, archetype: string) => Promise<KB2Result>;
  mood_analyzer: (messages: BaseMessage[]) => Promise<MoodAnalysis>;
  goal_evaluator: (goal: string) => Promise<GoalAnalysis>;
  progress_tracker: (user_id: string) => Promise<ProgressInsights>;
}
```

#### RAG System Setup

- **Vector Database**: Implement for each knowledge base
- **Embedding Strategy**: Semantic chunks with context preservation
- **Retrieval Enhancement**: Query expansion and context injection
- **Quality Scoring**: Relevance and accuracy metrics

### Backend Integrations

#### Enhanced Database Schema

```sql
-- Enhanced user profiles
ALTER TABLE users ADD COLUMN communication_preferences JSONB;
ALTER TABLE users ADD COLUMN coaching_history JSONB;
ALTER TABLE users ADD COLUMN success_patterns JSONB;

-- Session tracking
CREATE TABLE coaching_sessions (
  session_id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  session_goals JSONB,
  outcomes JSONB,
  insights_generated JSONB,
  tools_used JSONB[]
);

-- Tool usage analytics
CREATE TABLE tool_analytics (
  tool_name TEXT,
  usage_count INTEGER,
  success_rate FLOAT,
  avg_relevance_score FLOAT
);
```

#### API Integration Points

- **Knowledge Base APIs**: RAG query endpoints
- **Analytics APIs**: Tool usage and effectiveness tracking
- **User Profile APIs**: Enhanced profile management
- **Session APIs**: Comprehensive session tracking

---

## Conversation Flow Examples

### Example 1: Career Direction Confusion

**User Input**: "I'm feeling stuck in my career and don't know what direction to take."

**Brain Analysis**:

- Intent: Career guidance seeking
- Emotional state: Uncertainty/frustration
- Context needed: Current role, values, archetype strengths
- Strategy: Tool invocation for comprehensive analysis

**Hand Execution**:

- KB1 Query: "Career transition coaching approaches, values clarification"
- KB2 Query: "[User's archetype] + career development + strengths identification"
- Mood Analyzer: "stuck, uncertain, frustrated → needs confidence building"
- Goal Evaluator: "No clear goal yet → needs exploration phase"

**Mouth Synthesis**:
"I hear the frustration in feeling stuck - that's actually a really important signal that you're ready for something different. Many [archetype-specific insight] find that career clarity comes from understanding what energizes them most.

Let's start with what's making you feel most stuck right now - is it the work itself, the environment, or something about the growth opportunities?"

### Example 2: Leadership Challenge

**User Input**: "My team doesn't seem to respect my ideas, and I don't know how to get them to listen."

**Brain Analysis**:

- Intent: Leadership development need
- Challenge: Team dynamics and influence
- Context: Leadership style, team context, communication patterns
- Strategy: Multi-tool analysis for comprehensive guidance

**Hand Execution**:

- KB1 Query: "Building influence and credibility, team leadership"
- KB2 Query: "[User's archetype] + team leadership + communication style"
- KB3 Query: "Team respect and influence building scenarios"
- Progress Tracker: Check previous leadership discussions

**Mouth Synthesis**:
"That's a challenge many leaders face, especially when transitioning into more influence-based roles. The fact that you're aware of this dynamic shows strong leadership awareness.

What do you notice about when your ideas do land well versus when they don't? Is there a pattern in the types of ideas or how you're presenting them?"

---

## Success Metrics & Evaluation

### Agent Performance Metrics

1. **Response Relevance**: User feedback on response quality
2. **Tool Effectiveness**: Success rate of tool-generated insights
3. **Conversation Completion**: Users reaching meaningful outcomes
4. **Personalization Accuracy**: Archetype-specific guidance effectiveness
5. **Progress Tracking**: Long-term user development and goal achievement

### Knowledge Base Quality Metrics

1. **Retrieval Accuracy**: Relevance of retrieved information
2. **Coverage Completeness**: Ability to address diverse scenarios
3. **Update Frequency**: How often knowledge needs refreshing
4. **User Satisfaction**: Feedback on guidance quality

### System Performance Metrics

1. **Response Time**: Speed of complete conversation cycles
2. **Tool Coordination**: Efficiency of multi-tool orchestration
3. **Context Retention**: Accuracy of conversation continuity
4. **Error Recovery**: Handling of unclear or insufficient context

---

## Migration Strategy

### Phase 1: Infrastructure Setup (Weeks 1-2)

- Implement new state management system
- Set up RAG infrastructure for knowledge bases
- Create tool registry and integration framework
- Build basic 3-node conversation flow

### Phase 2: Tool Development (Weeks 3-4)

- Develop RAG systems for each knowledge base
- Implement assessment and analysis tools
- Create personalization engines
- Build tool coordination mechanisms

### Phase 3: Intelligence Integration (Weeks 5-6)

- Implement Brain node with advanced decision-making
- Develop Hand node tool orchestration
- Create Mouth node synthesis capabilities
- Integrate memory and preference systems

### Phase 4: Testing & Refinement (Weeks 7-8)

- Comprehensive conversation flow testing
- Tool effectiveness validation
- Performance optimization
- User feedback integration

### Phase 5: Deployment & Monitoring (Week 9)

- Production deployment with monitoring
- Performance metrics tracking
- User feedback collection
- Continuous improvement implementation

---

## Conclusion

This new 3-node AI agent architecture transforms the current linear coaching bot into a sophisticated, adaptive system that can:

1. **Intelligently analyze** user needs and determine optimal response strategies
2. **Dynamically utilize** multiple tools and knowledge sources for comprehensive context
3. **Synthesize personalized** coaching responses that drive meaningful progress

The architecture maintains the coaching excellence of the current system while adding the flexibility and intelligence needed for a truly adaptive AI coaching agent.

**Key advantages:**

- **Adaptability**: Responds to diverse user needs without rigid conversation flows
- **Intelligence**: Makes sophisticated decisions about information gathering and response strategy
- **Personalization**: Leverages multiple knowledge sources for highly customized coaching
- **Scalability**: Easily extensible with new tools and knowledge sources
- **Effectiveness**: Focused on driving real progress and meaningful outcomes

This design preserves the deep coaching expertise embedded in the current system while creating a foundation for advanced AI agent capabilities that can evolve with user needs and coaching best practices.
