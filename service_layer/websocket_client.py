import asyncio
import websockets
import json
from framework.common.logger.message_type import MessageType
from common.dto.websocket_conn.market_data_dto import  MarketDataDTO

class WebSocketClient:
    def __init__(self, ws_url, logger, data_callback):
        """Initializes the WebSocket client with a given URL, logger, and callback function."""
        self.ws_url = ws_url
        self.logger = logger
        self.data_callback = data_callback  # Function to send processed data
        self.connection = None
        self.is_connected = False

    async def connect(self):
        """Establishes a WebSocket connection and ensures continuous reconnection if it fails."""
        while not self.is_connected:
            try:
                self.logger.do_log(f"Connecting to WebSocket: {self.ws_url}...", MessageType.INFO)

                # Attempt connection with a 30-second timeout
                self.connection = await asyncio.wait_for(websockets.connect(self.ws_url), timeout=30)
                self.is_connected = True
                self.logger.do_log("Connection established.", MessageType.INFO)

                # Start listening for messages after a successful connection
                asyncio.create_task(self.listen())

            except asyncio.TimeoutError:
                self.logger.do_log(f"Connection to {self.ws_url} timed out. Retrying in 5 seconds...",
                                   MessageType.ERROR)
                await asyncio.sleep(5)  # Wait before retrying
            except Exception as e:
                self.logger.do_log(f"Failed to connect: {e}. Retrying in 5 seconds...", MessageType.ERROR)
                await asyncio.sleep(5)  # Wait before retrying

    async def disconnect(self):
        """Closes the WebSocket connection."""
        if self.connection:
            await self.connection.close()
            self.logger.do_log("WebSocket connection closed.", MessageType.INFO)
            self.is_connected = False

    async def listen(self):
        """Continuously listens for incoming messages and ensures automatic reconnection."""
        while True:  # Infinite loop to keep listening
            try:
                async for message in self.connection:
                    self.process_message(message)  # Process received message

            except websockets.exceptions.ConnectionClosed:
                self.logger.do_log("WebSocket connection lost. Reconnecting in 5 seconds...", MessageType.WARNING)
                self.is_connected = False
                await asyncio.sleep(5)  # Wait before retrying
                await self.connect()  # Attempt reconnection

            except Exception as e:
                self.logger.do_log(f"Unexpected error in listen(): {e}", MessageType.ERROR)
                await asyncio.sleep(5)  # Prevents an infinite loop of errors

    def process_message(self, message):
        """Parses the incoming message and converts it into MarketDataDTO."""
        try:
            data = json.loads(message)
            if "Msg" in data and data["Msg"] == "MarketDataMsg":
                market_data = MarketDataDTO(
                    symbol=data.get("Symbol", "UNKNOWN"),
                    exchange=data.get("Security", {}).get("Exchange", "UNKNOWN"),  # Maneja Exchange faltante
                    opening_price=data.get("OpeningPrice", 0.0),
                    high_price=data.get("TradingSessionHighPrice", 0.0),
                    low_price=data.get("TradingSessionLowPrice", 0.0),
                    closing_price=data.get("ClosingPrice", 0.0),
                    last_trade_price=data.get("Trade", 0.0),
                    trade_volume=data.get("TradeVolume", 0.0),
                    timestamp=data.get("MDEntryDate", "UNKNOWN")
                )
                self.data_callback(market_data)
        except Exception as e:
            self.logger.do_log(f"Error processing message: {e} - Message: {message}", MessageType.ERROR)

    def get_status(self):
        """Returns the current connection status (True if connected, False otherwise)."""
        return self.is_connected

