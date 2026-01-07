# persistence.py
import sqlite3
import json
from typing import Dict, Optional, List
from datetime import datetime
import threading
import os

class AgentPersistence:
    """Handles saving and loading agent state using SQLite with context support."""
    
    def __init__(self, db_path: str = "agent_memory.db"):
        """
        Initialize the persistence manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
            # Enable foreign keys
            self._local.conn.execute("PRAGMA foreign_keys = ON")
        return self._local.conn

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_conn() as conn:
            conn.executescript("""
                -- Agents table with context support
                CREATE TABLE IF NOT EXISTS agents (
                    name TEXT PRIMARY KEY,
                    persona TEXT,
                    instruction TEXT,
                    strategy TEXT,
                    context_collection TEXT,      -- Name of the ChromaDB collection
                    context_persist_dir TEXT,     -- Directory for context persistence
                    context_query TEXT,           -- Current context query
                    context_num_results INTEGER,  -- Number of results to retrieve
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Agent states table
                CREATE TABLE IF NOT EXISTS agent_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT,
                    task TEXT,
                    history TEXT,  -- JSON string of conversation history
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (agent_name) REFERENCES agents(name) ON DELETE CASCADE
                );
                
                -- Index for faster state lookups
                CREATE INDEX IF NOT EXISTS idx_agent_states_agent_name 
                ON agent_states(agent_name);
                
                -- Automatic timestamp update trigger
                CREATE TRIGGER IF NOT EXISTS update_agent_timestamp 
                AFTER UPDATE ON agents
                BEGIN
                    UPDATE agents 
                    SET last_updated = CURRENT_TIMESTAMP 
                    WHERE name = NEW.name;
                END;
            """)
    
    def save_agent_state(self, agent) -> bool:
        """
        Save agent's current state to database.
        
        Args:
            agent: The agent instance to save
            
        Returns:
            bool: True if successful
        """
        try:
            with self._get_conn() as conn:
                # Save base agent information with context details
                context_collection = None
                context_persist_dir = None
                context_query = None
                context_num_results = None
                
                if agent._context:
                    context_collection = agent._context.collection_name
                    context_persist_dir = agent._context.persist_dir
                    context_query = agent._context.current_query
                    # Default to 3 if not specified
                    context_num_results = getattr(agent._context, 'num_results', 3)
                
                # Update or insert agent record
                conn.execute("""
                    INSERT INTO agents (
                        name, persona, instruction, strategy,
                        context_collection, context_persist_dir,
                        context_query, context_num_results
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        persona = excluded.persona,
                        instruction = excluded.instruction,
                        strategy = excluded.strategy,
                        context_collection = excluded.context_collection,
                        context_persist_dir = excluded.context_persist_dir,
                        context_query = excluded.context_query,
                        context_num_results = excluded.context_num_results
                """, (
                    agent.name,
                    agent.persona,
                    agent.instruction,
                    agent.strategy.__class__.__name__ if agent.strategy else None,
                    context_collection,
                    context_persist_dir,
                    context_query,
                    context_num_results
                ))
                
                # Save current state with history
                history_json = json.dumps(agent.history)
                conn.execute("""
                    INSERT INTO agent_states (agent_name, task, history)
                    VALUES (?, ?, ?)
                """, (
                    agent.name,
                    agent.task,
                    history_json
                ))
                
            return True
        except Exception as e:
            print(f"Error saving agent state: {str(e)}")
            return False
    
    def load_agent_state(self, agent, agent_name: Optional[str] = None) -> bool:
        """
        Load agent's most recent state from database.
        
        Args:
            agent: The agent instance to update
            agent_name: Optional name of the agent state to load
            
        Returns:
            bool: True if successful
        """
        try:
            name_to_load = agent_name or agent.name
            with self._get_conn() as conn:
                # Load agent base information including context
                agent_data = conn.execute("""
                    SELECT 
                        persona, instruction, strategy,
                        context_collection, context_persist_dir,
                        context_query, context_num_results
                    FROM agents
                    WHERE name = ?
                """, (name_to_load,)).fetchone()
                
                if not agent_data:
                    return False
                
                # Load most recent state
                state_data = conn.execute("""
                    SELECT task, history
                    FROM agent_states
                    WHERE agent_name = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (name_to_load,)).fetchone()
                
                # Update agent with loaded data
                agent.persona = agent_data[0]
                agent.instruction = agent_data[1]
                if agent_data[2]:
                    agent.strategy = agent_data[2]
                
                # Initialize context if context data exists
                if agent_data[3] and agent_data[4]:  # context_collection and persist_dir
                    if not agent._context or agent._context.collection_name != agent_data[3]:
                        agent.initialize_context(
                            collection_name=agent_data[3],
                            persist_dir=agent_data[4]
                        )
                    if agent_data[5]:  # context_query
                        agent.set_context_query(
                            agent_data[5],
                            num_results=agent_data[6] or 3
                        )
                
                # Update task and history if state data exists
                if state_data:
                    agent.task = state_data[0]
                    agent._history = json.loads(state_data[1]) if state_data[1] else []
                
                return True
                
        except Exception as e:
            print(f"Error loading agent state: {str(e)}")
            return False
    
    def get_agent_history(self, agent_name: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve the last N states for an agent.
        
        Args:
            agent_name: Name of the agent
            limit: Maximum number of states to retrieve
            
        Returns:
            List[Dict]: List of historical states
        """
        try:
            with self._get_conn() as conn:
                states = conn.execute("""
                    SELECT task, history, timestamp
                    FROM agent_states
                    WHERE agent_name = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (agent_name, limit)).fetchall()
                
                return [{
                    'task': state[0],
                    'history': json.loads(state[1]) if state[1] else [],
                    'timestamp': state[2]
                } for state in states]
                
        except Exception as e:
            print(f"Error retrieving agent history: {str(e)}")
            return []
    
    def list_saved_agents(self) -> Dict[str, datetime]:
        """
        Return a dictionary of saved agent names and their last update times.
        
        Returns:
            Dict[str, datetime]: Dictionary mapping agent names to timestamps
        """
        saved_agents = {}
        try:
            with self._get_conn() as conn:
                results = conn.execute("""
                    SELECT name, last_updated
                    FROM agents
                    ORDER BY last_updated DESC
                """).fetchall()
                
                for name, timestamp in results:
                    saved_agents[name] = datetime.fromisoformat(timestamp)
                    
        except Exception as e:
            print(f"Error listing saved agents: {str(e)}")
            
        return saved_agents
    
    def delete_agent_state(self, agent_name: str) -> bool:
        """
        Delete all data for an agent.
        
        Args:
            agent_name: Name of the agent to delete
            
        Returns:
            bool: True if successful
        """
        try:
            with self._get_conn() as conn:
                # Due to foreign key constraints, this will also delete
                # all associated states
                conn.execute("DELETE FROM agents WHERE name = ?", (agent_name,))
                return True
        except Exception as e:
            print(f"Error deleting agent state: {str(e)}")
            return False
    
    def cleanup_old_states(self, agent_name: str, keep_last: int = 10) -> bool:
        """
        Clean up old states keeping only the N most recent ones.
        
        Args:
            agent_name: Name of the agent
            keep_last: Number of most recent states to keep
            
        Returns:
            bool: True if successful
        """
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    DELETE FROM agent_states 
                    WHERE agent_name = ? 
                    AND id NOT IN (
                        SELECT id FROM agent_states
                        WHERE agent_name = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    )
                """, (agent_name, agent_name, keep_last))
                return True
        except Exception as e:
            print(f"Error cleaning up old states: {str(e)}")
            return False