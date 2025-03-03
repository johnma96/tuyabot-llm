# This file serves as an example for configuring a runtime function.
# Here is the execution functions of the external behavior model trained in 2024-06

import sys
import pytz

from datetime import datetime
from tuyabot_llm import GetInfoWebPages, InfoToVectors

#--- Time zone for date handling
timezone_colombia = pytz.timezone('America/Bogota')


def run_get_original_data() -> None:
    """
    Retrieves the original data from the specified URLs.

    This function initializes a GetInfoWebPages object with the specified URLs and retrieves the original data.
    """
    urls = [
        "https://www.tuya.com.co/como-pago-mi-tarjeta-o-credicompras", 
        "https://www.tuya.com.co/tarjetas-de-credito", 
        "https://www.tuya.com.co/credicompras", 
        "https://www.tuya.com.co/otras-soluciones-financieras", 
        "https://www.tuya.com.co/nuestra-compania", 
        "https://www.tuya.com.co/activacion-tarjeta"
    ]

    get_info_pages_obj = GetInfoWebPages(urls=urls)


def run_process_org_data() -> None:
    """
    Processes the original data.

    This function initializes an InfoToVectors object and processes the original data.
    """
    info_to_vectors_obj = InfoToVectors()
    db = info_to_vectors_obj.process_original_info()


def run_create_data_bot() -> None:
    """
    Creates the data bot.

    This function calls the run_get_original_data and run_process_org_data functions to create the data bot.
    """
    run_get_original_data()
    run_process_org_data()


if __name__ == '__main__':
    """
    Main entry point of the script.

    This script accepts command-line arguments to select the process to run.
    """

    # Dictionary to select the process(function) to be executed
    process_to_run = {
        "make_data": run_create_data_bot,
        "get_original_data": run_get_original_data,
        "process_original_data": run_process_org_data
    }[sys.argv[1]]

    try:
        # Parse command-line arguments
        kwargs = {value.split('=')[0]:value.split('=')[1] for value in sys.argv[2:]}
    except:
        raise ValueError('A value that does not specify the parameter to configure has been entered. It should have the form key=value')
    
    # Record the start time of the process
    start_execution = datetime.now(timezone_colombia).strftime('%Y-%m-%d %H:%M:%S')
    print(f'-------- Process started at {start_execution} Colombian time --------\n')

    # Run the selected process
    process_to_run(**kwargs)