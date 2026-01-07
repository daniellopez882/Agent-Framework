import sqlite3
import json
from typing import Dict, Optional, List
from datetime import datetime
import threading

class AgentPersistence:
    """Handles saving and loading agent state using SQLite."""
    
    def __init__(self, db_path: str = "agent_memory.db"):
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
                CREATE TABLE IF NOT EXISTS agents (
                    name TEXT PRIMARY KEY,
                    persona TEXT,
                    instruction TEXT,
                    strategy TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS agent_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT,
                    task TEXT,
                    history TEXT,  -- Store JSON as TEXT
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (agent_name) REFERENCES agents(name) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_agent_states_agent_name 
                ON agent_states(agent_name);
                
                CREATE TRIGGER IF NOT EXISTS update_agent_timestamp 
                AFTER UPDATE ON agents
                BEGIN
                    UPDATE agents 
                    SET last_updated = CURRENT_TIMESTAMP 
                    WHERE name = NEW.name;
                END;
            """)
    
    def save_agent_state(self, agent) -> bool:
        """Save agent's current state to database."""
        try:
            with self._get_conn() as conn:
                # First, update or insert the agent's base information
                conn.execute("""
                    INSERT INTO agents (name, persona, instruction, strategy)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        persona = excluded.persona,
                        instruction = excluded.instruction,
                        strategy = excluded.strategy
                """, (
                    agent.name,
                    agent.persona,
                    agent.instruction,
                    agent.strategy.__class__.__name__ if agent.strategy else None
                ))
                
                # Convert history to JSON string
                history_json = json.dumps(agent.history)
                
                # Then, save the current state
                conn.execute("""
                    INSERT INTO agent_states (agent_name, task, history)
                    VALUES (?, ?, ?)
                """, (
                    agent.name,
                    agent.task,
                    history_json  # Save as JSON string
                ))
                
            return True
        except Exception as e:
            print(f"Error saving agent state: {str(e)}")
            return False
    
    def load_agent_state(self, agent, agent_name: Optional[str] = None) -> bool:
        """Load agent's most recent state from database."""
        try:
            name_to_load = agent_name or agent.name
            with self._get_conn() as conn:
                # Load agent base information
                agent_data = conn.execute("""
                    SELECT persona, instruction, strategy
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
                
                if not state_data:
                    return False
                
                # Update agent with loaded data
                agent.persona = agent_data[0]
                agent.instruction = agent_data[1]
                if agent_data[2]:
                    agent.strategy = agent_data[2]
                
                agent.task = state_data[0]
                # Parse JSON string back to list
                agent._history = json.loads(state_data[1]) if state_data[1] else []
                
                return True
                
        except Exception as e:
            print(f"Error loading agent state: {str(e)}")
            return False
    
    def get_agent_history(self, agent_name: str, limit: int = 10) -> List[Dict]:
        """Retrieve the last N states for an agent."""
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
        """Return a dictionary of saved agent names and their last update times."""
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
        """Delete all data for an agent."""
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
        """Clean up old states keeping only the N most recent ones."""
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