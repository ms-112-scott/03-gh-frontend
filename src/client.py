import asyncio
import websockets


async def hello():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        name = input("Enter your name: ")
        print(f"Client sending: {name}")
        await websocket.send(name)

        greeting = await websocket.recv()
        print(f"Client received: {greeting}")


if __name__ == "__main__":
    # 執行一次性的 Echo 測試
    asyncio.run(hello())
