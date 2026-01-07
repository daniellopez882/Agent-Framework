# Agent Framework 🤖

**Author: daniellopez882**

Agent Framework is a powerful Python foundation for building sophisticated, context-aware AI agents. It provides developers with the essential building blocks to create agents with distinct personas, structured reasoning, conversational memory, and robust tool integration.

## Key Features ✨

- **🎭 Persona Management**: Define rich identities, roles, and behavioral constraints for your AI agents.
- **🧠 Conversational Memory**: Maintain state and context across multi-turn interactions for seamless conversations.
- **🎯 Strategic Reasoning**: Empower agents to decompose complex requests into actionable plans using advanced reasoning patterns.
- **💾 State Persistence**: Built-in support for saving and restoring agent states, ensuring continuity across sessions.
- **📚 RAG-based Context**: Integrate Retrieval-Augmented Generation to provide agents with access to external knowledge bases.
- **🔧 Tool Extensibility**: Easily extend agent capabilities by connecting them to custom tools, APIs, and external services.

## Installation

### 1. Environment Setup
Create and activate a virtual environment to keep your dependencies isolated:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Getting Started

The framework is structured into modular components that demonstrate core agentic capabilities. You can explore the implementation across different complexity levels:

1. **Persona & Basics**: Basic agent configuration and personality definition.
2. **Memory Systems**: Implementing short-term and persistent memory.
3. **Reasoning Engines**: Enabling agents to plan and execute tasks.
4. **Context & Retrieval**: Adding RAG capabilities for knowledge-based tasks.
5. **Tool Integration**: Connecting agents to the real world through custom tools.

To run a basic flow, navigate to a component directory and execute the runner:

```bash
python flow.py
```

## Architecture

This framework is designed for extensibility. Each module handles a specific aspect of the agent's lifecycle, from initial prompt construction to tool execution and state management.

---

*Rebranded and maintained by daniellopez882.*