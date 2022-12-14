"""
This code is from transaction-classifier repository
https://github.com/banksalad/transaction-classifier/blob/master/classifier/domain/usecases/vectorizer.py
"""

from typing import List, Tuple

ASCII_LENGTH: int = 128

START_POINT_OF_HANGUL: int = ord('가')
END_POINT_OF_HANGUL: int = ord('힣')
START_POINT_OF_JAMO: int = ord('ㄱ')
END_POINT_OF_JAMO: int = ord('ㅣ')

CHOSEONG: str = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNGSEONG: str = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONGSEONG: List[str] = [
    '',
    'ㄱ',
    'ㄲ',
    'ㄱㅅ',
    'ㄴ',
    'ㄴㅈ',
    'ㄴㅎ',
    'ㄷ',
    'ㄹ',
    'ㄹㄱ',
    'ㄹㅁ',
    'ㄹㅂ',
    'ㄹㅅ',
    'ㄹㅌ',
    'ㄹㅍ',
    'ㄹㅎ',
    'ㅁ',
    'ㅂ',
    'ㅂㅅ',
    'ㅅ',
    'ㅆ',
    'ㅇ',
    'ㅈ',
    'ㅊ',
    'ㅋ',
    'ㅌ',
    'ㅍ',
    'ㅎ',
]

LEN_CHOSEONG: int = len(CHOSEONG)
LEN_JUNGSEONG: int = len(JUNGSEONG)
LEN_JONGSEONG: int = len(JONGSEONG)
HANGUL_LENGTH: int = LEN_CHOSEONG + LEN_JUNGSEONG + LEN_JONGSEONG

OOV_INDEX: int = 246
MAX_CHARACTER_VECTOR_LENGTH: int = 75
BANK_ACCOUNT_MAX_CHARACTER_VECTOR_LENGTH: int = 30


def get_vector_to_classify_category(
    features_to_be_vectorized: List[str],
) -> List[List[List[int]]]:
    return [
        [get_padded_character_vector(to_be_vectorized)]
        for to_be_vectorized in features_to_be_vectorized
    ]


def get_padded_character_vector(
    to_be_vectorized: str, max_vector_length: int = MAX_CHARACTER_VECTOR_LENGTH
) -> List[int]:
    vector = get_character_vector(to_be_vectorized)
    vector_length: int = len(vector)
    if vector_length >= max_vector_length:
        return vector[:max_vector_length]

    to_be_padded_length = max_vector_length - vector_length
    return vector + [OOV_INDEX] * to_be_padded_length


def get_character_vector(to_be_vectorized: str) -> List[int]:
    vectors: List[int] = []
    for char in to_be_vectorized:
        vectors.extend(
            get_indexes_of_decomposed_charactor(ordinalized_char=ord(char))
        )
    return vectors


def get_indexes_of_decomposed_charactor(ordinalized_char: int) -> List[int]:
    indexes: List[int] = []
    if START_POINT_OF_HANGUL <= ordinalized_char <= END_POINT_OF_HANGUL:
        indexes = get_indexes_of_decomposed_hangul(ordinalized_char)
    elif ordinalized_char < ASCII_LENGTH:
        indexes = get_indexes_of_ascii_code(ordinalized_char)
    elif START_POINT_OF_JAMO <= ordinalized_char <= END_POINT_OF_JAMO:
        indexes = get_indexes_of_decomposed_jamo(ordinalized_char)

    return indexes


def get_indexes_of_decomposed_hangul(ordinalized_char: int) -> List[int]:
    decomposed_indexes = []
    choseong, jungseong, jongseong = get_cho_jung_jong_offset(ordinalized_char)
    decomposed_indexes.append(choseong)
    decomposed_indexes.append(LEN_CHOSEONG + jungseong)
    if jongseong:
        decomposed_indexes.append(
            LEN_CHOSEONG + LEN_JUNGSEONG + (jongseong - 1)
        )

    return decomposed_indexes


def get_cho_jung_jong_offset(ordinalized_char: int) -> Tuple[int, int, int]:
    shifted_hangul = ordinalized_char - START_POINT_OF_HANGUL
    num_of_sets_of_cho_jung = shifted_hangul // LEN_JONGSEONG
    jongseong_offset = shifted_hangul % LEN_JONGSEONG
    choseong_offset = num_of_sets_of_cho_jung // LEN_JUNGSEONG
    jungseong_offset = num_of_sets_of_cho_jung % LEN_JUNGSEONG
    return choseong_offset, jungseong_offset, jongseong_offset


def get_indexes_of_ascii_code(ordinalized_char: int) -> List[int]:
    return [HANGUL_LENGTH + ordinalized_char]


def get_indexes_of_decomposed_jamo(ordinalized_char: int) -> List[int]:
    return [
        HANGUL_LENGTH + ASCII_LENGTH + (ordinalized_char - START_POINT_OF_JAMO)
    ]
