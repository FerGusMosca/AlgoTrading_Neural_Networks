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
        """Establishes a WebSocket connection."""
        try:
            self.logger.do_log(f"Connecting to WebSocket: {self.ws_url}...", MessageType.INFO)
            self.connection = await asyncio.wait_for(websockets.connect(self.ws_url), timeout=30)
            self.is_connected = True
            self.logger.do_log("Connection established.", MessageType.INFO)

            # Start listening for messages
            asyncio.create_task(self.listen())
        except Exception as e:
            self.logger.do_log(f"Failed to connect: {e}", MessageType.ERROR)
            self.is_connected = False

    async def disconnect(self):
        """Closes the WebSocket connection."""
        if self.connection:
            await self.connection.close()
            self.logger.do_log("WebSocket connection closed.", MessageType.INFO)
            self.is_connected = False

    async def listen(self):
        """Listens for incoming messages and processes market data."""
        try:
            async for message in self.connection:
                self.process_message(message)
        except websockets.exceptions.ConnectionClosed:
            self.logger.do_log("WebSocket connection lost.", MessageType.WARNING)
            self.is_connected = False

    def process_message(self, message):
        """Parses the incoming message and converts it into MarketDataDTO."""
        try:
            data = json.loads(message)
            if "Msg" in data and data["Msg"] == "MarketDataMsg":
                market_data = MarketDataDTO(
                    symbol=data["Symbol"],
                    exchange=data["BestBidExch"],
                    opening_price=data["OpeningPrice"],
                    high_price=data["TradingSessionHighPrice"],
                    low_price=data["TradingSessionLowPrice"],
                    closing_price=data["ClosingPrice"],
                    last_trade_price=data["Trade"],
                    timestamp=data["MDEntryDate"]
                )
                self.data_callback(market_data)
        except Exception as e:
            self.logger.do_log(f"Error processing message: {e}", MessageType.ERROR)

    def get_status(self):
        """Returns the current connection status (True if connected, False otherwise)."""
        return self.is_connected

