"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""

from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Literal,
    Optional,
)

from asgiref.sync import sync_to_async
from langchain_core.exceptions import LangChainException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import (
    END,
    StateGraph,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from psycopg_pool import AsyncConnectionPool

from app.core.config import (
    Environment,
    settings,
)
from app.core.langgraph.tools import tools
from app.core.logging import logger
from app.core.metrics import llm_inference_duration_seconds
from app.core.prompts import (
    BRAIN_PROMPT,
    SYNTHESIZER_PROMPT,
    SYSTEM_PROMPT,
)
from app.schemas import (
    BrainDecision,
    GraphState,
    Message,
    SynthesisResponse,
)
from app.utils import (
    dump_messages,
    prepare_messages,
)


class LangGraphAgent:
    """Manages the LangGraph Agent/workflow and interactions with the LLM.

    This class handles the creation and management of the LangGraph workflow,
    including LLM interactions, database connections, and response processing.
    """

    def __init__(self):
        """Initialize the LangGraph Agent with necessary components."""
        # Use environment-specific LLM model
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
            api_key=settings.LLM_API_KEY,
            max_tokens=settings.MAX_TOKENS,
            **self._get_model_kwargs(),
        ).bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None

        logger.info("llm_initialized", model=settings.LLM_MODEL, environment=settings.ENVIRONMENT.value)

    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get environment-specific model kwargs.

        Returns:
            Dict[str, Any]: Additional model arguments based on environment
        """
        model_kwargs = {}

        # Development - we can use lower speeds for cost savings
        if settings.ENVIRONMENT == Environment.DEVELOPMENT:
            model_kwargs["top_p"] = 0.8

        # Production - use higher quality settings
        elif settings.ENVIRONMENT == Environment.PRODUCTION:
            model_kwargs["top_p"] = 0.95
            model_kwargs["presence_penalty"] = 0.1
            model_kwargs["frequency_penalty"] = 0.1

        return model_kwargs

    async def _get_connection_pool(self) -> AsyncConnectionPool:
        """Get a PostgreSQL connection pool using environment-specific settings.

        Returns:
            AsyncConnectionPool: A connection pool for PostgreSQL database.
        """
        if self._connection_pool is None:
            try:
                # Configure pool size based on environment
                max_size = settings.POSTGRES_POOL_SIZE

                self._connection_pool = AsyncConnectionPool(
                    settings.POSTGRES_URL,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                    },
                )
                await self._connection_pool.open()
                logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
            except Exception as e:
                logger.error("connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we might want to degrade gracefully
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_connection_pool", environment=settings.ENVIRONMENT.value)
                    return None
                raise e
        return self._connection_pool

    async def _brain(self, state: GraphState) -> dict:
        """Brain node: Analyzes user request and plans what tools/context are needed.
        
        This node serves as the strategic planner that:
        1. Analyzes the user's request
        2. Determines what information is needed
        3. Decides which tools to call
        4. Plans the overall approach

        Args:
            state (GraphState): The current state of the conversation.

        Returns:
            dict: Updated state with brain's analysis and potential tool calls.
        """
        # Create structured LLM for brain decisions
        structured_llm = self.llm.with_structured_output(BrainDecision)
        
        # Create proper prompt template that maintains chat context and user archetype info
        archetype_context = ""
        if state.primary_archetype or state.secondary_archetype:
            archetype_context = f"\n\nUSER ARCHETYPE CONTEXT:\n"
            if state.primary_archetype:
                archetype_context += f"Primary Archetype: {state.primary_archetype}\n"
            if state.secondary_archetype:
                archetype_context += f"Secondary Archetype: {state.secondary_archetype}\n"
            if state.profile:
                archetype_context += f"Profile: {state.profile}\n"
        
        # Extract previous response strategy for context continuity
        previous_strategy = ""
        for msg in reversed(state.messages):
            if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                response_strategy = msg.additional_kwargs.get('response_strategy')
                if response_strategy:
                    previous_strategy = f"\n\nPREVIOUS RESPONSE'S STRATEGY:\nThis was the strategy used for the bot's last response: {response_strategy}\n"
                    break
        
        brain_prompt = ChatPromptTemplate.from_messages([
            ("system", BRAIN_PROMPT + archetype_context + previous_strategy),
            ("placeholder", "{messages}")  # This preserves the full chat context
        ])
        
        # Prepare the full conversation history - USE THE MESSAGES DIRECTLY
        # LangChain ChatPromptTemplate with placeholder handles message types correctly
        chat_messages = state.messages

        llm_calls_num = 0
        max_retries = settings.MAX_LLM_CALL_RETRIES
        current_model = settings.LLM_MODEL

        for attempt in range(max_retries):
            try:
                with llm_inference_duration_seconds.labels(model=current_model).time():
                    # Use structured chain with full context
                    chain = brain_prompt | structured_llm
                    brain_decision: BrainDecision = await chain.ainvoke({
                        "messages": chat_messages
                    })
                    
                    # Check if brain_decision is None (can happen with structured output)
                    if brain_decision is None:
                        raise LangChainException("LLM returned None for structured output")
                    
                # Create response message based on brain decision
                if brain_decision.direct_response:
                    # Brain provides direct response - include strategy for continuity
                    response_message = AIMessage(
                        content=brain_decision.direct_response,
                        additional_kwargs={"response_strategy": brain_decision.response_strategy}
                    )
                    generated_state = {"messages": [response_message]}
                elif brain_decision.needs_tools:
                    # Brain decides tools are needed - generate tool calls without content
                    # Use the same prompt structure to ensure ToolMessage context is preserved
                    tool_response = await self.llm.ainvoke(
                        brain_prompt.format_messages(messages=chat_messages)
                    )
                    
                    # Clean the tool response - remove content but keep tool_calls
                    if hasattr(tool_response, 'tool_calls') and tool_response.tool_calls:
                        clean_tool_message = AIMessage(
                            content="",  # Empty content - only tool calls
                            tool_calls=tool_response.tool_calls
                        )
                        generated_state = {"messages": [clean_tool_message]}
                    else:
                        # If no tool calls were generated, don't add any message
                        generated_state = {"messages": []}
                else:
                    # Brain decides synthesis is needed - DO NOT add message, let synthesizer handle it
                    generated_state = {"messages": []}
                
                # Store structured decision for routing
                generated_state["needs_context"] = brain_decision.needs_tools
                generated_state["brain_decision"] = brain_decision.model_dump()
                
                # Store response strategy for synthesizer if provided
                if brain_decision.response_strategy:
                    generated_state["response_strategy"] = brain_decision.response_strategy
                    
                logger.info(
                    "brain_analysis_completed",
                    session_id=state.session_id,
                    llm_calls_num=llm_calls_num + 1,
                    model=current_model,
                    complexity=brain_decision.complexity_level,
                    needs_tools=brain_decision.needs_tools,
                    needs_synthesis=brain_decision.needs_synthesis,
                    has_direct_response=bool(brain_decision.direct_response),
                    environment=settings.ENVIRONMENT.value,
                )
                return generated_state
            except LangChainException as e:
                logger.error(
                    "brain_analysis_failed",
                    llm_calls_num=llm_calls_num,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                    environment=settings.ENVIRONMENT.value,
                )
                llm_calls_num += 1

                # Fallback model in production
                if settings.ENVIRONMENT == Environment.PRODUCTION and attempt == max_retries - 2:
                    current_model = "gemini-2.0-flash"
                    logger.warning(
                        "using_fallback_model", model=current_model, environment=settings.ENVIRONMENT.value
                    )
                    self.llm = ChatGoogleGenerativeAI(
                        model=current_model,
                        temperature=settings.DEFAULT_LLM_TEMPERATURE,
                        api_key=settings.LLM_API_KEY,
                        max_tokens=settings.MAX_TOKENS,
                        **self._get_model_kwargs(),
                    ).bind_tools(tools)

                continue

        raise Exception(f"Failed to get brain analysis after {max_retries} attempts")

    async def _context_gathering(self, state: GraphState) -> dict:
        """Context gathering node: Executes all RAG retrieval and tool calls.
        
        This node:
        1. Executes tool calls from the brain node
        2. Performs RAG retrieval (when implemented)
        3. Gathers all necessary context
        4. Returns consolidated information

        Args:
            state: The current agent state containing messages and tool calls.

        Returns:
            dict: Updated state with gathered context and tool results.
        """
        last_message = state.messages[-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            logger.warning("context_gathering_called_without_tool_calls", session_id=state.session_id)
            return {"messages": []}
        
        outputs = []
        for tool_call in last_message.tool_calls:
            try:
                logger.info(
                    "executing_tool",
                    tool_name=tool_call["name"],
                    session_id=state.session_id
                )
                
                # Prepare tool arguments with user context for auth-required tools
                if tool_call["name"] in [
                    "fetch_user_commitments", 
                    "fetch_user_commitments_enhanced",
                    "create_user_commitment",
                    "complete_user_commitment",
                    "fetch_user_journal_entries", 
                    "set_user_commitment",
                    "fetch_user_reminders",
                    "set_user_reminder", 
                    "update_user_reminder",
                    "offer_commitment_reminder",
                    "set_commitment_reminder"
                ]:
                    logger.info(
                        "preparing_user_context",
                        tool_name=tool_call["name"],
                        has_user_id=bool(state.user_id),
                        has_access_token=bool(state.access_token),
                        user_id_length=len(state.user_id) if state.user_id else 0,
                        session_id=state.session_id
                    )
                    
                    # Call tool with injected user context and tool-specific arguments
                    tool_args = tool_call["args"].copy() if tool_call.get("args") else {}
                    tool_args.update({
                        "user_id": state.user_id,
                        "access_token": state.access_token
                    })
                    
                    # Add session_id as chat_id for commitment and reminder tools
                    if tool_call["name"] in ["set_user_commitment", "set_user_reminder"]:
                        tool_args["chat_id"] = state.session_id
                    
                    tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_args)
                else:
                    # For other tools, use normal invocation
                    tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=tool_result,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
                
                logger.info(
                    "tool_executed_successfully",
                    tool_name=tool_call["name"],
                    session_id=state.session_id
                )
            except Exception as e:
                logger.error(
                    "tool_execution_failed",
                    tool_name=tool_call["name"],
                    error=str(e),
                    session_id=state.session_id
                )
                outputs.append(
                    ToolMessage(
                        content=f"Error executing {tool_call['name']}: {str(e)}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        
        # TODO: Add RAG retrieval here when implemented
        # rag_results = await self._perform_rag_retrieval(state)
        # outputs.extend(rag_results)
        
        logger.info(
            "context_gathering_completed",
            session_id=state.session_id,
            tools_executed=len(outputs)
        )
        
        return {"messages": outputs}

    async def _synthesizer(self, state: GraphState) -> dict:
        """Synthesizer node: Creates final response using all gathered context.
        
        This node:
        1. Reviews the original user question
        2. Analyzes all gathered context and tool results
        3. Synthesizes information from multiple sources
        4. Generates the final comprehensive response

        Args:
            state: The current agent state with all context and messages.

        Returns:
            dict: Updated state with the final synthesized response.
        """
        # Create structured LLM for synthesis without tools
        synthesis_llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
            api_key=settings.LLM_API_KEY,
            max_tokens=settings.MAX_TOKENS,
            **self._get_model_kwargs(),
        ).with_structured_output(SynthesisResponse)
        
        # Create simplified prompt template that only uses strategy
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", SYNTHESIZER_PROMPT),
            ("user", "Original Question: {user_question}\n\nResponse Strategy: {response_strategy}")
        ])
        
        # Extract the latest user question
        user_question = "No specific question found"
        for msg in reversed(state.messages):
            if hasattr(msg, 'content') and hasattr(msg, 'role') and msg.role == 'user':
                user_question = msg.content
                break
        
        # Get the response strategy from brain
        response_strategy = state.response_strategy or "No strategy provided"

        llm_calls_num = 0
        max_retries = settings.MAX_LLM_CALL_RETRIES
        current_model = settings.LLM_MODEL

        for attempt in range(max_retries):
            try:
                with llm_inference_duration_seconds.labels(model=current_model).time():
                    # Use structured chain with strategy only
                    chain = synthesis_prompt | synthesis_llm
                    synthesis_result: SynthesisResponse = await chain.ainvoke({
                        "user_question": user_question,
                        "response_strategy": response_strategy
                    })
                    
                    # Check if synthesis_result is None (can happen with structured output)
                    if synthesis_result is None:
                        raise LangChainException("LLM returned None for structured output")
                    
                    # Create final response message with brain's strategy (not synthesizer's)
                    brain_strategy = state.response_strategy or "No strategy provided"
                    response_message = AIMessage(
                        content=synthesis_result.response,
                        additional_kwargs={"response_strategy": brain_strategy}
                    )
                    generated_state = {"messages": [response_message]}
                    
                logger.info(
                    "synthesis_completed",
                    session_id=state.session_id,
                    llm_calls_num=llm_calls_num + 1,
                    model=current_model,
                    has_strategy=bool(response_strategy and response_strategy != "No strategy provided"),
                    environment=settings.ENVIRONMENT.value,
                )
                return generated_state
            except LangChainException as e:
                logger.error(
                    "synthesis_failed",
                    llm_calls_num=llm_calls_num,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                    environment=settings.ENVIRONMENT.value,
                )
                llm_calls_num += 1

                # Fallback model in production
                if settings.ENVIRONMENT == Environment.PRODUCTION and attempt == max_retries - 2:
                    current_model = "gemini-2.0-flash"
                    logger.warning(
                        "using_fallback_model", model=current_model, environment=settings.ENVIRONMENT.value
                    )

                continue

        raise Exception(f"Failed to synthesize response after {max_retries} attempts")
    
    def _route_after_brain(self, state: GraphState) -> Literal["context_gathering", "synthesizer", "__end__"]:
        """Route after brain analysis: to context gathering, synthesis, or direct end.

        Args:
            state: The current agent state containing messages.

        Returns:
            Literal: 
            - "context_gathering" if tools needed
            - "synthesizer" if complex synthesis needed
            - "__end__" if brain can respond directly to simple queries
        """
        # Use structured brain decision if available
        brain_decision = getattr(state, 'brain_decision', None)
        if brain_decision:
            # Check if brain provided a direct response
            if brain_decision.get('direct_response'):
                return "__end__"
            
            # Check if tools are needed
            if brain_decision.get('needs_tools', False):
                # Verify we actually have tool calls before going to context gathering
                last_message = state.messages[-1] if state.messages else None
                if (last_message and 
                    hasattr(last_message, 'tool_calls') and 
                    last_message.tool_calls):
                    return "context_gathering"
                else:
                    # Brain said it needs tools but didn't provide any - go to synthesis instead
                    # This prevents infinite loops when brain can't decide on specific tools
                    return "synthesizer"
            
            # Check if synthesis is needed
            if brain_decision.get('needs_synthesis', True):
                return "synthesizer"
            
            # Default fallback
            return "__end__"
        
        # Fallback to original logic if no structured decision
        # Check if brain decided tools are needed (legacy compatibility)
        if state.needs_context:
            return "context_gathering"
        
        # Check if brain provided a direct response (no tool calls and has content)
        last_message = state.messages[-1]
        if (hasattr(last_message, 'content') and 
            last_message.content and 
            not getattr(last_message, 'tool_calls', None)):
            
            # Simple heuristic: if response is direct and doesn't mention needing more info
            content_lower = last_message.content.lower()
            synthesis_indicators = [
                "need to gather", "require additional", "let me search", 
                "need more information", "synthesize", "analyze further"
            ]
            
            # If no synthesis indicators found, brain can respond directly
            if not any(indicator in content_lower for indicator in synthesis_indicators):
                return "__end__"
        
        # Default to synthesizer for complex responses
        return "synthesizer"

    async def create_graph(self) -> Optional[CompiledStateGraph]:
        """Create and configure the LangGraph workflow.

        Returns:
            Optional[CompiledStateGraph]: The configured LangGraph instance or None if init fails
        """
        if self._graph is None:
            try:
                graph_builder = StateGraph(GraphState)
                # Add the three main nodes: brain -> context_gathering -> synthesizer
                graph_builder.add_node("brain", self._brain)
                graph_builder.add_node("context_gathering", self._context_gathering)
                graph_builder.add_node("synthesizer", self._synthesizer)
                
                # Set up the flow: brain -> (context_gathering OR synthesizer OR end)
                graph_builder.add_conditional_edges(
                    "brain",
                    self._route_after_brain,
                    {
                        "context_gathering": "context_gathering", 
                        "synthesizer": "synthesizer",
                        "__end__": END
                    },
                )
                graph_builder.add_edge("context_gathering", "brain")
                graph_builder.add_edge("synthesizer", END)
                
                # Set entry point
                graph_builder.set_entry_point("brain")

                # Get connection pool (may be None in production if DB unavailable)
                connection_pool = await self._get_connection_pool()
                if connection_pool:
                    checkpointer = AsyncPostgresSaver(connection_pool)
                    await checkpointer.setup()
                else:
                    # In production, proceed without checkpointer if needed
                    checkpointer = None
                    if settings.ENVIRONMENT != Environment.PRODUCTION:
                        raise Exception("Connection pool initialization failed")

                self._graph = graph_builder.compile(
                    checkpointer=checkpointer, name=f"{settings.PROJECT_NAME} Agent ({settings.ENVIRONMENT.value})"
                )

                logger.info(
                    "graph_created",
                    graph_name=f"{settings.PROJECT_NAME} Agent",
                    environment=settings.ENVIRONMENT.value,
                    has_checkpointer=checkpointer is not None,
                )
            except Exception as e:
                logger.error("graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we don't want to crash the app
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_graph")
                    return None
                raise e

        return self._graph

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        access_token: Optional[str] = None,
        profile: Optional[str] = None,
        primary_archetype: Optional[str] = None,
        secondary_archetype: Optional[str] = None,
    ) -> tuple[list[dict], dict]:
        """Get a response from the LLM with notification payload extraction.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for Langfuse tracking.
            user_id (Optional[str]): The user ID for Langfuse tracking.

        Returns:
            tuple[list[dict], dict]: The response messages and notification payload.
        """
        if self._graph is None:
            self._graph = await self.create_graph()
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [CallbackHandler()],
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": False,
            },
        }
        try:
            graph_state = {
                "messages": dump_messages(messages), 
                "session_id": session_id,
                "user_id": user_id,
                "access_token": access_token,
                "profile": profile or "",
                "primary_archetype": primary_archetype or "",
                "secondary_archetype": secondary_archetype or "",
            }
            
            logger.info(
                "graph_state_created",
                session_id=session_id,
                has_user_id=bool(user_id),
                has_access_token=bool(access_token),
                user_id=user_id,
                access_token_length=len(access_token) if access_token else 0
            )
            
            response = await self._graph.ainvoke(graph_state, config)
            processed_messages = self.__process_messages(response["messages"])
            notification_payload = self.__extract_notification_payload(response["messages"])
            return processed_messages, notification_payload
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            raise e

    async def get_stream_response(
        self, messages: list[Message], session_id: str, user_id: Optional[str] = None,
        access_token: Optional[str] = None,
        profile: Optional[str] = None,
        primary_archetype: Optional[str] = None,
        secondary_archetype: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.

        Yields:
            str: Tokens of the LLM response.
        """
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [
                CallbackHandler(
                    environment=settings.ENVIRONMENT.value, debug=False, user_id=user_id, session_id=session_id
                )
            ],
        }
        if self._graph is None:
            self._graph = await self.create_graph()

        try:
            graph_state = {
                "messages": dump_messages(messages), 
                "session_id": session_id,
                "user_id": user_id,
                "access_token": access_token,
                "profile": profile or "",
                "primary_archetype": primary_archetype or "",
                "secondary_archetype": secondary_archetype or "",
            }
            async for token, _ in self._graph.astream(
                graph_state, config, stream_mode="messages"
            ):
                try:
                    yield token.content
                except Exception as token_error:
                    logger.error("Error processing token", error=str(token_error), session_id=session_id)
                    # Continue with next token even if current one fails
                    continue
        except Exception as stream_error:
            logger.error("Error in stream processing", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        if self._graph is None:
            self._graph = await self.create_graph()

        state: StateSnapshot = await sync_to_async(self._graph.get_state)(
            config={"configurable": {"thread_id": session_id}}
        )
        return self.__process_messages(state.values["messages"]) if state.values else []

    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        processed_messages = []
        openai_style_messages = convert_to_openai_messages(messages)
        
        # keep just assistant and user messages with actual content (exclude tool-only messages)
        for i, message in enumerate(openai_style_messages):
            # Skip messages with empty or whitespace-only content before creating Message objects
            if not message.get('content') or not message.get('content').strip():
                continue
                
            # For assistant messages, check if original message has response_strategy in additional_kwargs
            if message.get('role') == 'assistant' and i < len(messages):
                original_msg = messages[i]
                if hasattr(original_msg, 'additional_kwargs') and original_msg.additional_kwargs:
                    response_strategy = original_msg.additional_kwargs.get('response_strategy')
                    if response_strategy:
                        message['response_strategy'] = response_strategy
            
            # Only create Message objects for messages with content that passes validation
            if message.get('role') in ["assistant", "user"] and message.get('content') and message.get('content').strip():
                processed_messages.append(Message(**message))
        
        return processed_messages
    
    def __extract_notification_payload(self, messages: list[BaseMessage]) -> dict:
        """Extract notification payload from tool response messages.
        
        Args:
            messages: List of messages to search for notification payloads
            
        Returns:
            dict: Notification payload data or empty dict
        """
        import json
        import re
        
        for message in reversed(messages):  # Check latest messages first
            if hasattr(message, 'content') and message.content:
                # Look for notification payload patterns
                payload_match = re.search(r'\[NOTIFICATION_PAYLOAD: (.+?)\]', message.content)
                if payload_match:
                    try:
                        payload_data = eval(payload_match.group(1))  # Safe eval of dict
                        
                        # Validate that this message actually confirms a reminder was set
                        if self.__validate_payload_legitimacy(message.content, payload_data):
                            logger.info(
                                "notification_payload_extracted",
                                payload=payload_data
                            )
                            return payload_data
                        else:
                            logger.info(
                                "notification_payload_rejected",
                                reason="Message does not confirm reminder was actually set"
                            )
                            return {}
                    except Exception as e:
                        logger.warning(
                            "notification_payload_parse_error",
                            error=str(e),
                            raw_payload=payload_match.group(1)
                        )
        
        return {}  # Return empty dict if no payload found
    
    def __validate_payload_legitimacy(self, message_content: str, payload_data: dict) -> bool:
        """Validate that the message actually confirms a reminder was set.
        
        Args:
            message_content: The full message content
            payload_data: The extracted payload data
            
        Returns:
            bool: True if payload is legitimate, False otherwise
        """
        # Check for confirmation keywords that indicate reminder was actually set
        confirmation_patterns = [
            r"✅.*Perfect.*reminder.*set",
            r"✅.*reminder.*successfully.*set",
            r"Reminder.*successfully.*set",
            r"Perfect.*I've.*set.*up.*reminder",
            r"reminder.*set.*up.*successfully"
        ]
        
        import re

        # Must have confirmation language in the message
        has_confirmation = any(
            re.search(pattern, message_content, re.IGNORECASE) 
            for pattern in confirmation_patterns
        )
        
        if not has_confirmation:
            logger.info(
                "payload_validation_failed",
                reason="No confirmation language found",
                message_preview=message_content[:100] + "..." if len(message_content) > 100 else message_content
            )
            return False
        
        # Must have required payload fields for a valid reminder
        required_fields = ["should_schedule", "reminder_type"]
        has_required_fields = all(field in payload_data for field in required_fields)
        
        if not has_required_fields:
            logger.info(
                "payload_validation_failed",
                reason="Missing required fields",
                payload=payload_data
            )
            return False
        
        # should_schedule must be True for legitimate reminders
        if not payload_data.get("should_schedule", False):
            logger.info(
                "payload_validation_failed",
                reason="should_schedule is not True",
                payload=payload_data
            )
            return False
        
        logger.info(
            "payload_validation_passed",
            confirmation_found=has_confirmation,
            payload=payload_data
        )
        return True

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        try:
            # Make sure the pool is initialized in the current event loop
            conn_pool = await self._get_connection_pool()

            # Use a new connection for this specific operation
            async with conn_pool.connection() as conn:
                for table in settings.CHECKPOINT_TABLES:
                    try:
                        await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                        logger.info(f"Cleared {table} for session {session_id}")
                    except Exception as e:
                        logger.error(f"Error clearing {table}", error=str(e))
                        raise

        except Exception as e:
            logger.error("Failed to clear chat history", error=str(e))
            raise
