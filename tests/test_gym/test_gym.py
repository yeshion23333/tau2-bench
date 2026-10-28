from unittest.mock import Mock

import pytest

from tau2.data_model.message import UserMessage
from tau2.gym.gym_agent import AgentGymEnv, GymAgent, TauSpace
from tau2.user.user_simulator import UserSimulator

from .utils import timeout


class TestTauGymEnv:
    """Test cases for TauGymEnv."""

    def test_tau_gym_env_initialization(self):
        """Test that TauGymEnv initializes correctly."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        assert env.domain == "mock"
        assert env.task_id == "create_task_1"
        assert env._orchestrator is None
        assert env._agent is None
        assert env._user is None
        assert not env._simulation_done.is_set()
        assert isinstance(env.observation_space, TauSpace)
        assert isinstance(env.action_space, TauSpace)

    @timeout(10)
    def test_tau_gym_env_reset(self):
        """Test that TauGymEnv reset works correctly."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Test reset
        observation, info = env.reset()

        # Check that reset returns expected types
        assert isinstance(observation, str)
        assert isinstance(info, dict)

        # Check that orchestrator and agent are initialized
        assert env._orchestrator is not None
        assert env._agent is not None
        assert env._user is not None
        assert isinstance(env._agent, GymAgent)

        # Check that the orchestrator thread is running
        assert env._orchestrator_thread is not None
        assert env._orchestrator_thread.is_alive()

    @timeout(15)
    def test_tau_gym_env_step(self):
        """Test that TauGymEnv step works correctly."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Reset the environment first
        observation, info = env.reset()

        # Test step with an action
        action = "Hello! I'm here to help you. What can I assist you with today?"
        observation, reward, terminated, truncated, info = env.step(action)

        # Check that step returns expected types
        assert isinstance(observation, str)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Check that the action was processed
        assert len(observation) > 0

    @timeout(20)
    def test_tau_gym_env_multiple_steps(self):
        """Test that TauGymEnv can handle multiple steps."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Reset the environment
        observation, info = env.reset()

        # Take multiple steps
        actions = [
            "Hello! I'm here to help you.",
            "I understand you need assistance.",
            "Let me guide you through the process.",
        ]

        for i, action in enumerate(actions):
            observation, reward, terminated, truncated, info = env.step(action)

            # Check that we get a valid observation
            assert isinstance(observation, str)
            assert isinstance(terminated, bool)

            # If terminated, break
            if terminated:
                break

        # Check that we can take at least one step
        assert len(actions) > 0

    def test_tau_gym_env_step_without_reset(self):
        """Test that TauGymEnv raises an error when step is called without reset."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Try to step without reset
        with pytest.raises(RuntimeError, match="Orchestrator not initialized"):
            env.step("test action")

    @timeout(15)
    def test_tau_gym_env_reset_multiple_times(self):
        """Test that TauGymEnv can be reset multiple times."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Reset multiple times
        for i in range(3):
            observation, info = env.reset()

            # Check that reset works each time
            assert isinstance(observation, str)
            assert isinstance(info, dict)
            assert env._orchestrator is not None
            assert env._agent is not None

    @timeout(15)
    def test_tau_gym_env_observation_format(self):
        """Test that TauGymEnv formats observations correctly."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Reset the environment
        observation, info = env.reset()

        # Test step to get an observation
        action = "Hello! How can I help you?"
        observation, reward, terminated, truncated, info = env.step(action)

        # Check that observation is a string
        assert isinstance(observation, str)

        # If observation is not empty, it should contain role: content format
        if observation:
            # Check that it contains at least one role: content pattern
            lines = observation.split("\n")
            for line in lines:
                if line.strip():  # Skip empty lines
                    assert ":" in line, f"Line '{line}' should contain ':'"

    @pytest.mark.parametrize("defer_user_stop", [False, True])
    @pytest.mark.parametrize(
        "token", ["###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"]
    )
    @pytest.mark.parametrize("ending", ["done", "max_steps", "agent_error"])
    def test_tau_gym_env_termination(self, monkeypatch, defer_user_stop, token, ending):
        """User STOP may close dialogue while business tools finish before cleanup."""
        replies = iter(
            [
                "Create Important Meeting for user_1.",
                f"Yes, please proceed. {token}",
            ]
        )
        user_generate = Mock(
            side_effect=lambda message, state: (
                UserMessage(role="user", content=next(replies)),
                state,
            )
        )
        monkeypatch.setattr(UserSimulator, "generate_next_message", user_generate)
        monkeypatch.setattr(AgentGymEnv, "_get_reward", lambda self: (0.0, "{}"))
        env = AgentGymEnv(
            domain="mock", task_id="create_task_1", defer_user_stop=defer_user_stop
        )
        try:
            env.reset()
            observation, _, terminated, _, _ = env.step("May I create that task?")
            assert token in observation
            assert user_generate.call_count == 2
            if not defer_user_stop or token != "###STOP###":
                assert terminated
                assert env._simulation_run.termination_reason == "user_stop"
                return

            assert not terminated
            assert env._orchestrator.user_dialogue_closed
            assert env._simulation_run is None
            observation, _, terminated, _, _ = env.step("Please confirm again.")
            assert observation.startswith("Error: User dialogue is closed.")
            assert not terminated
            assert user_generate.call_count == 2
            observation, _, terminated, _, _ = env.step(
                '{"name":"create_task","arguments":{"user_id":"user_1","title":"Important Meeting"}}'
            )
            assert "Important Meeting" in observation
            assert not terminated
            assert any(
                task.title == "Important Meeting"
                for task in env._orchestrator.environment.tools.db.tasks.values()
            )
            if ending == "done":
                _, _, terminated, _, _ = env.step('{"name":"done","arguments":{}}')
                assert terminated
                expected_reason = "agent_stop"
            else:
                env.close(termination_reason=ending)
                expected_reason = ending
            assert env._simulation_run.termination_reason == expected_reason
            assert (
                len([m for m in env._simulation_run.messages if m.role == "user"]) == 2
            )
            assert any(
                call.name == "create_task"
                for message in env._simulation_run.messages
                for call in (getattr(message, "tool_calls", None) or [])
            )
            if ending != "done":
                assert not any(
                    m.content == "###STOP###" for m in env._simulation_run.messages
                )
        finally:
            env.close()
        assert not env._orchestrator_thread.is_alive()

    @timeout(15)
    def test_tau_gym_env_thread_safety(self):
        """Test that TauGymEnv handles threading correctly."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Reset the environment
        observation, info = env.reset()

        # Check that the orchestrator thread is running
        assert env._orchestrator_thread.is_alive()

        # Take a step
        action = "Test action"
        observation, reward, terminated, truncated, info = env.step(action)

        # Check that the thread is still running (unless terminated)
        if not terminated:
            assert env._orchestrator_thread.is_alive()

    def test_tau_gym_env_invalid_domain(self):
        """Test that TauGymEnv handles invalid domain gracefully."""
        with pytest.raises(Exception):  # Should raise some exception for invalid domain
            env = AgentGymEnv(domain="invalid_domain", task_id="create_task_1")
            env.reset()

    def test_tau_gym_env_invalid_task_id(self):
        """Test that TauGymEnv handles invalid task_id gracefully."""
        with pytest.raises(ValueError):  # Should raise ValueError for invalid task_id
            env = AgentGymEnv(domain="mock", task_id="invalid_task_id")
            env.reset()

    def test_tau_gym_env_reset_simple(self):
        """Test that TauGymEnv reset creates components correctly without running full simulation."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Test that components can be created
        environment = env._get_environment()
        task = env._get_task()
        agent = env._get_agent()
        user = env._get_user()
        orchestrator = env._get_orchestrator()

        # Check that all components are created successfully
        assert environment is not None
        assert task is not None
        assert agent is not None
        assert user is not None
        assert orchestrator is not None
        assert task.id == "create_task_1"

        # Check that the agent is a GymAgent
        assert isinstance(agent, GymAgent)

        print("✅ Simple reset test completed successfully!")

    def test_tau_gym_env_basic_functionality(self):
        """Test basic TauGymEnv functionality without running full simulation."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Test initialization
        assert env.domain == "mock"
        assert env.task_id == "create_task_1"
        assert env._orchestrator is None

        # Test that we can create the components
        environment = env._get_environment()
        task = env._get_task()
        agent = env._get_agent()
        user = env._get_user()
        orchestrator = env._get_orchestrator()

        # Check that all components are created successfully
        assert environment is not None
        assert task is not None
        assert agent is not None
        assert user is not None
        assert orchestrator is not None
        assert task.id == "create_task_1"

        print("✅ Basic functionality test passed!")


if __name__ == "__main__":
    # Run a simple test with timeout
    @timeout(30)
    def run_basic_test():
        env = AgentGymEnv(domain="mock", task_id="create_task_1")
        observation, info = env.reset()
        print(f"Initial observation: {observation}")

        action = "Hello! I'm here to help you."
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"After step observation: {observation}")
        print(f"Terminated: {terminated}")

        print("✅ TauGymEnv basic test completed!")

    # Run the test
    result = run_basic_test()
    if result is None:
        print("⚠️ Test timed out - this is expected for a full simulation")
