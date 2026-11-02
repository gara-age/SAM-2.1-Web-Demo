"""Small, explicit Korean command aliases; not general language translation."""
import re

NOUNS={'사람':'person','사람들':'person','남자':'man','여자':'woman','컵':'cup','병':'bottle',
       '의자':'chair','책상':'desk','책':'book','노트북':'laptop','키보드':'keyboard',
       '마우스':'computer mouse','휴대폰':'cell phone','강아지':'dog','개':'dog','고양이':'cat',
       '자동차':'car','가방':'bag','모자':'hat','안경':'glasses','손':'hand','얼굴':'face'}
COLORS={'빨간':'red','빨강':'red','붉은':'red','파란':'blue','파랑':'blue','초록':'green',
        '녹색':'green','노란':'yellow','검은':'black','검정':'black','흰':'white','하얀':'white'}


def concept_prompt(text):
    original=text.strip()
    if not original or len(original)>120:raise ValueError('대상을 1~120자로 입력하세요.')
    text=re.sub(r'(?:을|를)?\s*(?:선택해(?:줘)?|선택|찾아줘|찾아|추적해(?:줘)?|추적)\s*[.!?]*$','',original).strip()
    text=re.sub(r'^모든\s*','',text).strip()
    text=re.sub(r'\s+',' ',text)
    if not re.search('[가-힣]',text):
        text=re.sub(r'^(?:select|find|track|segment)\s+(?:the\s+)?','',text,flags=re.I).strip()
        if not text:raise ValueError('선택할 대상 이름을 입력하세요.')
        return text
    compact=text.replace(' ','')
    color=''
    for word,value in COLORS.items():
        if compact.startswith(word):color=value+' ';compact=compact[len(word):];break
    if compact in NOUNS:return color+NOUNS[compact]
    raise ValueError('이 한국어 표현은 아직 자동 변환하지 못합니다. person, red cup처럼 영어 대상 표현을 입력하세요.')
