from itertools import product

MAX_CHARACTER_VECTOR_LENGTH = 300
ASCII_LENGTH: int = 128
START_POINT_OF_HANGUL: int = ord('가')
END_POINT_OF_HANGUL: int = ord('힣')
START_POINT_OF_JAMO: int = ord('ㄱ')
END_POINT_OF_JAMO: int = ord('ㅣ')

CHOSEONG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNGSEONG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONGSEONG = [
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

유형_to_num_map = {'사실형': 0, '추론형': 1, '예측형': 2, '대화형': 3}
극성_to_num_map = {'긍정': 0, '부정': 1, '미정': 2}
시제_to_num_map = {'현재': 0, '과거': 1, '미래': 2}
확실성_to_num_map = {'확실': 0, '불확실': 1}
num_to_유형_map = {v: k for k, v in 유형_to_num_map.items()}
num_to_극성_map = {v: k for k, v in 극성_to_num_map.items()}
num_to_시제_map = {v: k for k, v in 시제_to_num_map.items()}
num_to_확실성_map = {v: k for k, v in 확실성_to_num_map.items()}

full_name_label_to_num_map = dict()
for i, (유형, 극성, 시제, 확실성) in enumerate(
        product(
            ['사실형', '추론형', '예측형', '대화형'],
            ['긍정', '부정', '미정'],
            ['현재', '과거', '미래'],
            ['확실', '불확실']
        )
):
    label = "-".join([유형, 극성, 시제, 확실성])
    full_name_label_to_num_map[label] = i
num_to_full_name_label_map = {v: k for k, v in full_name_label_to_num_map.items()}
