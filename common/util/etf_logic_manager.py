from common.dto.etf_position_dto import ETFPositionDTO
from common.util.csv_reader import CSVReader


class ETFLogicManager():


    @staticmethod
    def __extract_etf_composition__(etf_path,symbol_col,weights_col):
        symbols_csv=CSVReader.extract_col_csv(etf_path, symbol_col)
        weights_csv = CSVReader.extract_col_csv(etf_path, weights_col)

        etf_comp_dto_arr= ETFPositionDTO.build_positions_dto_arr(symbols_csv, weights_csv)
        ETFPositionDTO.validate_weights(etf_comp_dto_arr)

        return etf_comp_dto_arr