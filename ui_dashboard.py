import time

from common.util.logging.logger import Logger
from common.util.std_in_out.ml_settings_loader import MLSettingsLoader
from controllers.main_dashboard_controller import MainDashboardController



if __name__ == '__main__':
    loader = MLSettingsLoader()
    logger = Logger()

    # loader user to load settings
    config_settings = loader.load_settings("./configs/commands_mgr.ini")

    main_dash_contr = MainDashboardController(logger, config_settings)
    main_dash_contr.display()
    print(f"Main Dashboard successfully shown...")
    # 🛑 Block the main thread to prevent script from exiting
    try:
        while True:
            time.sleep(1)  # Sleep to avoid CPU overuse
    except KeyboardInterrupt:
        print("Shutting down gracefully...")