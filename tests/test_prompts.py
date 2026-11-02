import pytest
from backend.prompts import concept_prompt


@pytest.mark.parametrize('text,expected',[('사람 선택해','person'),('사람을 선택해줘','person'),
    ('빨간 컵 선택해','red cup'),('모든 의자 찾아줘','chair'),('person','person'),
    ('select the red cup','red cup'),('person wearing a striped shirt','person wearing a striped shirt')])
def test_explicit_aliases_and_open_english_phrases(text,expected):
    assert concept_prompt(text)==expected


def test_unsupported_korean_is_not_silently_mistranslated():
    with pytest.raises(ValueError):concept_prompt('선생님 뒤에 있는 사람')
    with pytest.raises(ValueError):concept_prompt('')
