from multi_agent_lab.core.message import Message
from multi_agent_lab.core.message_bus import MessageBus


async def test_publish_delivers_message_to_receiver() -> None:
    bus = MessageBus()
    inbox = await bus.subscribe("coder")
    message = Message(
        sender="planner", receiver="coder", type="task.created", content={"task_id": "1"}
    )

    await bus.publish(message)

    received = await inbox.get()
    assert received == message


async def test_publish_delivers_message_to_broadcast_subscriber() -> None:
    bus = MessageBus()
    inbox = await bus.subscribe("*")
    message = Message(sender="planner", receiver="coder", type="task.created", content={})

    await bus.publish(message)

    received = await inbox.get()
    assert received.receiver == "coder"
    assert received.type == "task.created"
