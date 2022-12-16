from typing import List, Tuple
from sentence_classification.constants import (
    MAX_CHARACTER_VECTOR_LENGTH,
    ASCII_LENGTH,
    START_POINT_OF_HANGUL,
    END_POINT_OF_HANGUL,
    START_POINT_OF_JAMO,
    END_POINT_OF_JAMO,
    CHOSEONG,
    JUNGSEONG,
    LEN_CHOSEONG,
    LEN_JUNGSEONG,
    LEN_JONGSEONG,
    HANGUL_LENGTH,
    OOV_INDEX
)


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
