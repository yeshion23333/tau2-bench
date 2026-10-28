import json

import pytest

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.user.user_simulator import StopValidationError, UserSimulator


@pytest.fixture
def user_instructions() -> str:
    return (
        "You are Mia Li. You want to fly from New York to Seattle on May 20 (one way)."
    )


@pytest.fixture
def bad_user_instructions() -> str:
    return "You are Mia Li. You want to fly from Chicago to San Francisco on May 19 (round trip)."


@pytest.fixture
def first_agent_message() -> AssistantMessage:
    return AssistantMessage(
        content="Hello, how can I help you today?", role="assistant"
    )


@pytest.fixture
def user_simulator(user_instructions: str) -> UserSimulator:
    return UserSimulator(llm="gpt-4o-mini", instructions=user_instructions)


@pytest.mark.parametrize(
    "responses,expected",
    [
        (["Please book it."], "Please book it."),
        (["###TRANSFER###"], "###TRANSFER###"),
        (
            ["Yes, please book it.###STOP###", '{"decision":"continue"}'],
            "Yes, please book it.",
        ),
        (
            ["Thanks, all done.###STOP###", '{"decision":"stop"}'],
            "Thanks, all done.###STOP###",
        ),
        (["I give up.###STOP###", '{"decision":"stop"}'], "I give up.###STOP###"),
        (
            ["Maybe.###STOP###", '{"decision":"uncertain"}', "Please book it."],
            "Please book it.",
        ),
        (
            [
                "Maybe.###STOP###",
                "invalid JSON",
                "Yes.###STOP###",
                '{"decision":"continue"}',
            ],
            "Yes.",
        ),
        (
            ["###STOP###", '{"decision":"continue"}', "I want to book."],
            "I want to book.",
        ),
        (
            [
                "Maybe.###STOP###",
                '{"decision":"uncertain"}',
                "Maybe.###STOP###",
                '{"decision":"uncertain"}',
            ],
            None,
        ),
    ],
)
def test_user_simulator(
    user_simulator: UserSimulator,
    first_agent_message: AssistantMessage,
    monkeypatch,
    responses,
    expected,
):
    calls = []
    remaining = iter(responses)

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if kwargs["call_name"] == "user_stop_validation":
            assert kwargs.get("tools") is None
            payload = json.loads(kwargs["messages"][1].content)
            assert payload["history"] == [
                {"role": "assistant", "content": first_agent_message.content}
            ]
            assert user_simulator.instructions not in kwargs["messages"][1].content
        else:
            assert kwargs["messages"][-1].content == first_agent_message.content
            assert len(kwargs["messages"]) == 2  # Same context on regeneration.
        return AssistantMessage(
            role="assistant",
            content=next(remaining),
            cost=1.0,
            usage={"prompt_tokens": 10, "completion_tokens": 2},
            raw_data={"original_response_preserved": True},
        )

    monkeypatch.setattr("tau2.user.user_simulator.generate", fake_generate)
    user_state = user_simulator.get_init_state()
    assert user_state is not None
    if expected is None:
        with pytest.raises(StopValidationError) as exc:
            user_simulator.generate_next_message(first_agent_message, user_state)
        assert len(exc.value.records) == 2
        assert all(row["decision"] == "uncertain" for row in exc.value.records)
        assert user_state.messages == [first_agent_message]
        assert len(calls) == 4
        return
    user_msg, user_state = user_simulator.generate_next_message(
        first_agent_message, user_state
    )
    # Check the response is a user message
    assert isinstance(user_msg, UserMessage)
    # Check the state is updated
    assert user_state is not None
    # Check the messages are of the correct type
    assert isinstance(user_state.messages[0], AssistantMessage)
    assert user_state.messages[0].content == first_agent_message.content
    assert isinstance(user_state.messages[1], UserMessage)
    assert user_msg.content == expected
    assert len(user_state.messages) == 2
    assert user_msg.cost == len(calls)
    assert user_msg.usage == {
        "prompt_tokens": len(calls) * 10,
        "completion_tokens": len(calls) * 2,
    }
    assert len(calls) == len(responses)
    assert user_msg.raw_data["original_response_preserved"]
    records = user_msg.raw_data.get("stop_validation")
    if "###STOP###" in responses[0]:
        assert records[0]["raw_reply"] == responses[0]
        assert records[-1]["filtered_reply"] == expected
        assert user_msg.model_dump()["raw_data"]["stop_validation"] == records
    else:
        assert records is None


def test_user_simulator_set_state(
    user_simulator: UserSimulator,
):
    user_simulator.get_init_state(
        message_history=[
            UserMessage(content="Hello, can you help me find a flight?", role="user"),
            AssistantMessage(
                content="Hello, I can help you find a flight.", role="assistant"
            ),
        ]
    )
