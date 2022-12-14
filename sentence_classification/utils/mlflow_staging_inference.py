import asyncio
import json
import time
import timeit

import numpy as np
import requests
import torch
from aiohttp import ClientSession

from laboratory.transaction_classification.utils.vectorizer import (
    get_padded_character_vector,
)


def get_one_sample():
    to_be_vectorized = "Sample Title"
    title_vec_inputs = get_padded_character_vector(to_be_vectorized, 75)
    title_vec_inputs = np.array(title_vec_inputs, dtype=np.int64)
    title_vec_inputs = torch.from_numpy(title_vec_inputs)
    tabular_vec_inputs = torch.randn((18)).tolist()
    data = {
        "inputs": [
            {
                "title_vec": title_vec_inputs.tolist(),
                "transaction_amount_vec": tabular_vec_inputs,
            }
        ]
    }
    return data


def get_inference_samples(n_samples=2):
    to_be_vectorized = "Sample Title"
    title_vec_inputs = get_padded_character_vector(to_be_vectorized, 75)
    title_vec_inputs = np.array(title_vec_inputs, dtype=np.int64)
    title_vec_inputs = torch.from_numpy(title_vec_inputs)
    title_vec_inputs = torch.stack(
        [title_vec_inputs for i in range(n_samples)]
    ).tolist()
    tabular_vec_inputs = torch.randn((n_samples, 18)).tolist()
    data = {
        "inputs": {
            "title_vec": title_vec_inputs,
            "transaction_amount_vec": tabular_vec_inputs,
        }
    }
    return data


def request_to_mlflow(n_samples=2):
    data = get_inference_samples(n_samples=n_samples)
    headers = {'Content-Type': 'application/json'}
    URL = "http://localhost:8082/v2/electronic-payment-transaction/classification/predict"
    res = requests.post(URL, data=json.dumps(data), headers=headers)
    print(res.json())


async def request_to_mlflow_wtih_aiohttp(
    session: ClientSession,
    data,
    headers={'Content-Type': 'application/json'},
    proxy=None,
    timeout=10,
    url="http://localhost:8082/v2/electronic-payment-transaction/classification/predict",
):
    response = await session.post(
        url=url, data=data, headers=headers, proxy=proxy, timeout=timeout
    )
    response_json = None
    try:
        response_json = await response.json(content_type=None)
    except json.decoder.JSONDecodeError as e:
        pass

    response_content = None
    try:
        response_content = await response.read()
    except:
        pass

    return (response.status, response_json, response_content)


async def parallel_request_to_mlflow(
    session: ClientSession, num_requests, n_samples
):
    sample_data_list = [
        get_inference_samples(n_samples=n_samples) for i in range(num_requests)
    ]
    t1 = time.time()
    results = await asyncio.gather(
        *[
            request_to_mlflow_wtih_aiohttp(session, json.dumps(data))
            for data in sample_data_list
        ]
    )
    t2 = time.time()
    t = t2 - t1
    return results, t


async def main():
    session = ClientSession()
    for num_transactions in (10, 100, 200):
        for num_parallel_requests in (3, 5, 10):
            elapsed_time_list = []
            for i in range(10):
                results, elapsed_time = await parallel_request_to_mlflow(
                    session,
                    num_requests=num_parallel_requests,
                    n_samples=num_transactions,
                )
                elapsed_time_list.append(elapsed_time)
                for result in results:
                    assert result[0] == 200, "Test Failed!"
            print(
                f'num_transactions: {num_transactions}, '
                f'num_parallel_requests: {num_parallel_requests}, '
                f'number_of_total_transaction_request: {num_transactions * num_parallel_requests}, '
                f'mean_elapsed_time: {np.mean(elapsed_time_list): .2f} s'
            )


asyncio.run(main())
# print(timeit.timeit(request_to_mlflow, number=100))
