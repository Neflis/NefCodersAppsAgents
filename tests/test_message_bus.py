from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus


async def test_publish_delivers_message_to_event_subscribers() -> None:
    bus = MessageBus()
    first_inbox = await bus.subscribe(EventType.TASK_CREATED)
    second_inbox = await bus.subscribe(EventType.TASK_CREATED)
    message = Message(sender="planner", type=EventType.TASK_CREATED, content={"task_id": "1"})

    await bus.publish(message)

    assert await first_inbox.get() == message
    assert await second_inbox.get() == message


async def test_publish_delivers_message_to_wildcard_subscriber() -> None:
    bus = MessageBus()
    inbox = await bus.subscribe("*")
    message = Message(sender="planner", type=EventType.TASK_CREATED, content={})

    await bus.publish(message)

    received = await inbox.get()
    assert received.type == EventType.TASK_CREATED


async def test_correlation_and_causation_are_preserved() -> None:
    bus = MessageBus()
    inbox = await bus.subscribe(EventType.CODE_PROPOSED)
    root = Message(sender="demo", type=EventType.GOAL_SUBMITTED, content={})
    child = Message(
        sender="coder",
        type=EventType.CODE_PROPOSED,
        content={},
        correlation_id=root.correlation_id,
        causation_id=root.id,
    )

    await bus.publish(child)

    received = await inbox.get()
    assert received.correlation_id == root.correlation_id
    assert received.causation_id == root.id
