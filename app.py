import os
import streamlit as st
import pandas as pd
from io import BytesIO
import json
import time

# --------------------------- LangChain ---------------------------
from langchain_unstructured import UnstructuredLoader
from langchain.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
# ---------------------------------------------------------------

###############################################################################
# 0. OpenAI 클라이언트 초기화 & SYSTEM_PROMPT
###############################################################################
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
if not OPENAI_API_KEY:
    st.error("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 확인하세요.")
    st.stop()

SYSTEM_PROMPT = """한국의 초등학교 2022 개정 교육과정 전문가입니다.
학교자율시간 계획서를 다음 원칙에 따라 작성합니다:

1. 지도계획에 모든 차시에 학습내용과 학습 주제가 빈틈없이 내용이 꼭 들어가야 합니다.
2. {grades}의 특성에 맞게 작성하도록 합니다. 
3. 학습자 중심의 교육과정 구성
4. 실생활 연계 및 체험 중심 활동
5. 교과 간 연계 및 통합적 접근
6. 과정 중심 평가와 피드백 강조
7. 유의미한 학습경험 제공
8. 요구사항을 반영한 맞춤형 교육과정 구성
9. 교수학습 방법의 다양화
10. 객관적이고 공정한 평가계획 수립
"""

###############################################################################
# --------------------------- 추가 기능(사이드바 타이핑) -----------------------
###############################################################################
def sidebar_typewriter_effect(text, delay=0.001):
    """사이드바에 한 글자씩 타이핑되듯 출력"""
    placeholder = st.sidebar.empty()
    output = ""
    for char in text:
        output += char
        placeholder.markdown(output)
        time.sleep(delay)
    return output

###############################################################################
# 1. 페이지 기본 설정 (스타일)
###############################################################################
def set_page_config():
    try:
        st.set_page_config(page_title="학교자율시간 계획서 생성기", page_icon="📚", layout="wide")
    except Exception as e:
        st.error(f"페이지 설정 오류: {e}")

    st.markdown("""
    <style>
    .main .block-container {
        padding: 2rem;
        max-width: 1200px;
        font-size: 1rem;
        line-height: 1.5;
    }
    .step-header {
        background-color: #f1f5f9;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin: 1.5rem 0 1rem;
        border-left: 4px solid #3b82f6;
    }
    .step-header h3 {
        margin: 0;
        font-size: 1.25rem;
    }
    .step-container-outer {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        padding: 10px 20px;
    }
    .step-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 1rem 0;
        flex-direction: row;
        width: 100%;
        padding: 20px;
        gap: 0.5rem;
    }
    .step-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        z-index: 2;
    }
    .step-circle {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-bottom: 8px;
        transition: all 0.3s ease;
    }
    .step-active {
        background-color: #3b82f6;
        color: white;
        box-shadow: 0 0 10px rgba(59,130,246,0.6);
        transform: scale(1.1);
    }
    .step-completed {
        background-color: #10b981;
        color: white;
    }
    .step-pending {
        background-color: #e5e7eb;
        color: #6b7280;
    }
    .step-label {
        font-size: 0.9rem;
        color: #374151;
        text-align: center;
        margin-top: 4px;
        width: 6rem;
        white-space: nowrap;
    }
    .step-line {
        height: 4px;
        flex: 1;
        background-color: #e5e7eb;
        margin: 0 10px;
        position: relative;
        top: -24px;
        z-index: 1;
        transition: background-color 0.3s ease;
    }
    .step-line-completed {
        background-color: #10b981;
    }
    .step-line-active {
        background-color: #3b82f6;
    }
    .stForm {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .stForm label {
        font-weight: 600;
    }
    button[kind="primary"] {
        border-radius: 4px;
        transition: background-color 0.2s ease;
    }
    button[kind="primary"]:hover {
        background-color: #2563eb !important;
    }
    .stTabs [role="tablist"] .stTabButton {
        background-color: #f1f5f9 !important;
        border: 1px solid #e5e7eb !important;
        border-bottom: none !important;
        color: #1f2937 !important;
        font-weight: 500 !important;
    }
    .stTabs [role="tablist"] .stTabButton[data-baseweb="tab"]:hover {
        background-color: #e2e8f0 !important;
    }
    .stTabs [role="tablist"] .stTabButton[data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff !important;
        border-top: 3px solid #3b82f6 !important;
        color: #1f2937 !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .sidebar-questions button {
        margin-bottom: 0.5rem;
        text-align: left;
        background: #f1f5f9 !important;
        color: #111827 !important;
        width: 100%;
        border: 1px solid #e5e7eb;
    }
    .sidebar-questions button:hover {
        background: #e2e8f0 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def show_progress():
    current_step = st.session_state.get('step', 1)
    steps = ["기본정보", "승인 신청서 다운로드", "내용체계", "성취기준", "교수학습 및 평가", "차시별계획", "최종 검토"]

    html = '<div class="step-container-outer"><div class="step-container">'
    for i, step_label in enumerate(steps, 1):
        if i < current_step:
            circle_class = "step-completed"
            icon = "✓"
        elif i == current_step:
            circle_class = "step-active"
            icon = str(i)
        else:
            circle_class = "step-pending"
            icon = str(i)
        html += f'''
            <div class="step-item">
                <div class="step-circle {circle_class}">{icon}</div>
                <div class="step-label">{step_label}</div>
            </div>
        '''
        if i < len(steps):
            if i < current_step:
                line_style = "step-line-completed"
            elif i == current_step:
                line_style = "step-line-active"
            else:
                line_style = ""
            html += f'<div class="step-line {line_style}"></div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)


STEP_DOC_MAP = {
    1: "step1_info",
    3: "step3_contents",
    4: "step4_standards",
    5: "step5_teaching",
    6: "step6_plan"
}

def load_document(file_path: str):
    lower_fp = file_path.lower()
    if lower_fp.endswith(".pdf"):
        return UnstructuredPDFLoader(file_path).load()
    elif lower_fp.endswith(".docx"):
        return UnstructuredWordDocumentLoader(file_path).load()
    else:
        return UnstructuredLoader(file_path).load()

@st.cache_resource
def get_vector_store_for_step(step: int):
    """
    1) step not in STEP_DOC_MAP -> None
    2) 해당 단계의 인덱스 폴더에서만 FAISS.load_local() 시도
    3) 실패(폴더 미존재/에러) 시 None 반환
    """
    if step not in STEP_DOC_MAP:
        return None

    doc_filename = STEP_DOC_MAP[step]
    base_name = os.path.splitext(doc_filename)[0]
    index_dir = f"faiss_index_{base_name}"

    if not os.path.isdir(index_dir):
        st.warning(f"단계 {step}에 대한 인덱스 폴더({index_dir})가 없습니다.")
        return None

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception as e:
        st.error(f"[ERROR] 단계 {step}: 인덱스 로딩 실패. (사유: {e})")
        return None


@st.cache_resource
def setup_combined_vector_store():
    """
    챗봇: documents 폴더 내 모든 PDF/DOCX/TXT를 한 번에 인덱싱 -> faiss_index/
    """
    index_dir = "faiss_index"
    if os.path.exists(index_dir) and os.path.isdir(index_dir):
        st.success("챗봇: 기존 통합 인덱스 로딩")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        st.info("챗봇: 통합 인덱스가 없어, 문서를 로딩 후 임베딩합니다.")
        docs_folder = "./documents/"
        exts = (".pdf",".docx",".txt")
        all_docs = []
        for fname in os.listdir(docs_folder):
            if fname.lower().endswith(exts):
                path = os.path.join(docs_folder,fname)
                try:
                    loaded = load_document(path)
                    all_docs.extend(loaded)
                except:
                    pass
        if not all_docs:
            st.warning("챗봇: documents 폴더 내 문서가 없어 인덱스 생성 불가")
            return None

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vs = FAISS.from_documents(all_docs, embeddings)
        vs.save_local(index_dir)
        st.success("챗봇: 통합 인덱스 생성 완료")
        return vs


def make_code_prefix(grades, subjects, activity_name):
    grade_part = ""
    if grades:
        grade_part = grades[0].replace("학년","").replace("학년군","").strip()
    subject_part = ""
    if subjects:
        subject_part = subjects[0][0] if subjects[0] else ""
    act_part = ""
    if activity_name:
        act_part = activity_name[:2]
    return f"{grade_part}{subject_part}{act_part}"

def generate_content(step, data, vector_store):
    """
    단계별 문서 검색 -> context -> 프롬프트 -> JSON parsing
    2/6/7단계는 문서 없음 -> {}
    """
    if step in [2,7]:
        return {}

    context = ""
    if step >= 3 and vector_store:
        query_map = {
            3: "내용체계",
            4: "성취기준",
            5: "교수학습 및 평가"
        }
        query = query_map.get(step,"")
        if query:
            retriever = vector_store.as_retriever()
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([d.page_content for d in docs])

    necessity = data.get('necessity','')
    overview = data.get('overview','')
    content_sets = data.get("content_sets",[])
    standards = data.get("standards",[])
    num_sets = len(content_sets)

    # 여기는 샘플 프롬프트들
    step_prompts = {
        1: f"""학교자율시간 활동의 기본 정보를 작성해주세요.

활동명: {data.get('activity_name')}
요구사항: {data.get('requirements')}
학교급: {data.get('school_type')}
대상 학년: {', '.join(data.get('grades', []))}
연계 교과: {', '.join(data.get('subjects', []))}
총 차시: {data.get('total_hours')}차시, 주당 {data.get('weekly_hours')}차시
운영 학기: {', '.join(data.get('semester', []))}

아래 예시와 같이, 주어진 **활동명**에 종속되어 결과물이 도출되도록 
'필요성(necessity)', '개요(overview)'만 작성해 주세요.

지침
1. 필요성은 예시의 2~3배 분량으로 작성해주세요.
2. 개요는 괄호( )로 목적·목표·주요 내용을 구분해 주세요

[예시]
필요성:
 • 불확실한 미래사회를 살아갈 학생들에게 필수적 요소인 디지털 기기의 바른 이해와 사용법에 대한 학습이 필요
 • 디지털 기기 활용뿐 아니라 디지털 윤리에 관한 학습을 통해 디지털 리터러시와 책임감 있는 디지털 시민으로서의 역량 함양 필요

개요:
 <목적>
 • 디지털 기기 사용 경험을 바탕으로, 디지털 기술의 원리와 활용, 윤리적 문제점을 탐구하며 안전하고 책임감 있는 디지털 시민으로 성장
 <목표>
 • 디지털 기기의 작동 원리와 활용 방법을 이해한다.
 • 디지털 기기를 안전하고 책임감 있게 사용하는 방법을 익힌다.
 <주요 내용>
 • 디지털 기기 작동 원리 및 간단한 프로그래밍
 • 디지털 기기를 활용한 다양한 창작 활동
 • 디지털 윤리에 대한 이해와 실천

다음 JSON 형식으로 작성 (성격은 제외):
{{
  "necessity": "작성된 필요성 내용",
  "overview": "작성된 개요 내용"
}}
""",

        3: f"""
이전 단계 결과:
활동명: {data.get('activity_name')}
요구사항: {data.get('requirements')}
학교급: {data.get('school_type')}
대상 학년: {', '.join(data.get('grades', []))}
연계 교과: {', '.join(data.get('subjects', []))}
이전 단계 결과를 참고하여 작성하기
아래 예시를 참고하여,
 핵심 아이디어는 IB교육에서 말하는 빅아이디어처럼, 학생들이 도달할 수 있는 일반화된 이론을 예시처럼 문장으로 진술.
'영역명(domain)', '핵심 아이디어(key_ideas)', '내용 요소(content_elements)'(지식·이해 / 과정·기능 / 가치·태도) 4개 세트를 JSON으로 작성.
'content_elements'에는 **'knowledge_and_understanding'(지식·이해), 'process_and_skills'(과정·기능), 'values_and_attitudes'(가치·태도)** 필수.
예시:
영역명
 기후위기와 기후행동

핵심 아이디어
 • 인간은 여러 활동을 통해 기후변화를 초래하였고, 기후변화는 우리의 삶에 다방면으로 영향을 미친다.
 • 우리는 직면한 기후변화 문제를 완화하거나 적응함으로써 대처하며 생활 속에서 자신이 실천할 수 있는 방법을 탐색하고 행동해야 한다.

내용 요소
 -지식·이해
  • 기후변화와 우리 삶의 관계
  • 기후변화와 식생활
 -과정·기능
  • 의사소통 및 갈등해결
  • 창의적 문제해결
 -가치·태도
  • 환경 공동체의식
  • 환경 실천
{context}
JSON 배열로 4개 객체.
[
  {{
    "domain": "...",
    "key_ideas": [...],
    "content_elements": {{
      "knowledge_and_understanding": [...],
      "process_and_skills": [...],
      "values_and_attitudes": [...]
    }}
  }},
  ...
]
""",

        4: f"""{context}
이전 단계
활동명: {data.get('activity_name')}
요구사항: {data.get('requirements')}
학교급: {data.get('school_type')}
대상 학년: {', '.join(data.get('grades', []))}
연계 교과: {', '.join(data.get('subjects', []))} 
내용 체계: {content_sets}

총 {num_sets}개 내용체계 세트가 생성되었으므로, 성취기준도 {num_sets}개 생성.

아래는 학년/교과/활동명에서 추출한 코드 접두사:
code_prefix: "{make_code_prefix(data.get('grades', []), data.get('subjects', []), data.get('activity_name',''))}"

지침:
1. 성취기준코드는 반드시 code_prefix에 -01, -02, ... 식으로 순서 붙여 생성.
2. 성취기준은 내용체계표 내용과 비슷하게. 문장 형식 예시:
   [4사세계시민-01] 글을 읽고 지구촌의 여러 문제를 이해하고 생각한다.
3. 성취기준 levels는 A/B/C (상/중/하) 세 단계 작성.
JSON 예시:
[
  {{
    "code": "code_prefix-01",
    "description": "성취기준 설명",
    "levels": [
      {{ "level": "A", "description": "상 수준 설명" }},
      {{ "level": "B", "description": "중 수준 설명" }},
      {{ "level": "C", "description": "하 수준 설명" }}
    ]
  }},
  ...
]
""",

        5: f"""{context}
이전 단계(성취기준): {standards}

(5단계) 교수학습 및 평가를 아래 예시 형식으로 작성해 주세요.
특히 "teaching_methods_text"를 **여러 줄**로 작성하되, 예시처럼 구체적인 활동 안내나 유의사항이 들어가도록 합니다.

[작성 지침]
1. 평가는 '평가요소', '수업평가방법', '평가기준'으로 구분
2. 평가기준은 상·중·하로 나누어 각각 작성
3. "teaching_methods_text"에는 **구체적인 수업 절차나 유의사항**을 2~3개 정도 항목으로 작성
4. **예시 형식**을 최대한 비슷하게 따르세요. 불필요한 문장(예: "아래와 같습니다" 등)은 넣지 말고, **JSON**만 반환

[교수학습방법 예시]
teaching_methods_text 예시:
- 학생들이 자신의 생활 속 경험을 바탕으로 다양한 문제상황을 탐색하도록 유도
- 활동 시에는 협동학습, 토의토론, 프로젝트학습 등 다양한 방법을 활용
- 학습 전 안전교육을 실시하고, 수업 중 안전수칙 준수 여부를 확인

[평가 예시]
평가요소
 • 국가유산의 의미와 유형 알아보고 가치 탐색하기

수업평가방법
 [개념학습/프로젝트]
 • 국가유산의 의미를 이해한 후, 국가유산을 유형별로 분류하고 가치 파악하기

평가기준
 • 상: 국가유산의 의미와 유형을 정확히 이해하며, 조사한 내용을 정리하여 가치를 구체적으로 설명할 수 있다.
 • 중: 국가유산의 의미와 유형을 이해하고, 조사한 내용을 통해 가치를 간단히 설명할 수 있다.
 • 하: 주변의 도움을 받아 국가유산의 의미와 유형을 간단히 말할 수 있다.

"teaching_methods_text": 문자열 (여러 줄 가능),
"assessment_plan": [
  {{
    "code": "성취기준코드",
    "description": "성취기준문장",
    "element": "평가요소",
    "method": "수업평가방법",
    "criteria_high": "상 수준 평가기준",
    "criteria_mid": "중 수준 평가기준",
    "criteria_low": "하 수준 평가기준"
  }},
  ...
]
"""
    }

    if step not in step_prompts:
        return {}

    prompt = step_prompts[step]
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt + "\n\n(위 형식으로 JSON만 반환)")
    ]
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o",  
        temperature=0.7,
        max_tokens=1800
    )
    response = chat(messages)
    raw_text = response.content.strip().replace("```json","").replace("```","").strip()

    try:
        parsed = json.loads(raw_text)
        return parsed
    except:
        st.warning(f"JSON 파싱 실패.\n원문:\n{raw_text}")
        return {}


def show_step_1(vector_store):
    st.markdown("<div class='step-header'><h3>1단계: 기본 정보</h3></div>", unsafe_allow_html=True)

    # 초기값
    if "school_type" not in st.session_state.data:
        st.session_state.data["school_type"] = "초등학교"
    if "grades" not in st.session_state.data:
        st.session_state.data["grades"] = []
    if "subjects" not in st.session_state.data:
        st.session_state.data["subjects"] = []

    current_school_type = st.session_state.data.get('school_type', '초등학교')

    # 학교급 바꾸기 버튼
    if st.button("학교급 바꾸기", use_container_width=True):
        if current_school_type == "초등학교":
            st.session_state.data["school_type"] = "중학교"
        else:
            st.session_state.data["school_type"] = "초등학교"
        st.session_state.data["grades"] = []
        st.session_state.data["subjects"] = []
        st.session_state.step = 1
        st.rerun()

    if 'generated_step_1' not in st.session_state:
        with st.form("basic_info_form"):
            options = ["초등학교", "중학교"]
            idx = 0 if st.session_state.data["school_type"] == "초등학교" else 1
            school_type = st.radio("학교급", options, index=idx)

            total_hours = st.number_input("총 차시", min_value=1, max_value=68,
                                          value=st.session_state.data.get('total_hours',29),
                                          help="총 차시 입력")
            semester = st.multiselect("운영 학기", ["1학기","2학기"],
                                      default=st.session_state.data.get('semester',["1학기"]))

            st.markdown("#### 학년 선택")
            if school_type=="초등학교":
                grades = st.multiselect("학년", ["3학년","4학년","5학년","6학년"],
                                        default=st.session_state.data.get('grades',[]))
                subjects = st.multiselect("교과",
                                          ["국어","수학","사회","과학","영어","음악","미술","체육","실과","도덕"],
                                          default=st.session_state.data.get('subjects',[]))
            else:
                grades = st.multiselect("학년", ["1학년","2학년","3학년"],
                                        default=st.session_state.data.get('grades',[]))
                subjects = st.multiselect("교과",
                                          ["국어","수학","사회/역사","과학/기술","영어","음악","미술","체육","정보","도덕"],
                                          default=st.session_state.data.get('subjects',[]))

            activity_name = st.text_input("활동명",
                                          value=st.session_state.data.get('activity_name',''),
                                          placeholder="예: 인공지능 놀이터")
            requirements = st.text_area("요구사항",
                                        value=st.session_state.data.get('requirements',''),
                                        placeholder="예: 디지털 리터러시 강화 필요",
                                        height=100)
            weekly_hours = st.number_input("주당 차시", min_value=1, max_value=5,
                                           value=st.session_state.data.get('weekly_hours',1),
                                           help="주당 몇 차시씩 운영되는지")
            submit_btn = st.form_submit_button("정보 생성 및 다음 단계로", use_container_width=True)

        if submit_btn:
            if activity_name and requirements and grades and subjects and semester:
                with st.spinner("정보 생성 중..."):
                    st.session_state.data["school_type"] = school_type
                    st.session_state.data["grades"] = grades
                    st.session_state.data["subjects"] = subjects
                    st.session_state.data["activity_name"] = activity_name
                    st.session_state.data["requirements"] = requirements
                    st.session_state.data["total_hours"] = total_hours
                    st.session_state.data["weekly_hours"] = weekly_hours
                    st.session_state.data["semester"] = semester

                    # 1단계 - 필요성, 개요
                    info = generate_content(1, st.session_state.data, vector_store)
                    if info:
                        st.session_state.data.update(info)
                        st.success("기본 정보 생성 완료.")
                        st.session_state.generated_step_1 = True
            else:
                st.error("모든 항목을 입력해주세요.")
    else:
        # 이미 생성됨 -> 수정
        with st.form("edit_basic_info_form"):
            st.markdown("#### 생성된 내용 수정")
            necessity = st.text_area("활동의 필요성",
                                     value=st.session_state.data.get('necessity',''),
                                     height=150)
            overview = st.text_area("활동 개요",
                                    value=st.session_state.data.get('overview',''),
                                    height=150)
            save_btn = st.form_submit_button("수정사항 저장 및 다음 단계로", use_container_width=True)

        if save_btn:
            with st.spinner("수정사항 저장 중..."):
                st.session_state.data["necessity"] = necessity
                st.session_state.data["overview"] = overview
                del st.session_state.generated_step_1
                st.success("수정사항 저장 완료.")
                st.session_state.step = 2
                st.rerun()

def show_step_2_approval(vector_store):
    st.markdown("<div class='step-header'><h3>2단계: 자율시간 승인 신청서 다운로드</h3></div>", unsafe_allow_html=True)
    st.info("입력한 기본 정보를 바탕으로 승인 신청서 엑셀 파일을 생성합니다.")

    fields = ["학교급", "대상 학년", "총 차시", "운영 학기", "연계 교과", "활동명", "요구사항", "필요성", "개요"]
    selected_fields = st.multiselect("다운로드할 항목 선택:", fields, fields,
                                     help="원하는 항목만 선택하여 파일에 포함할 수 있습니다.")

    if selected_fields:
        excel_data = create_approval_excel_document(selected_fields)
        st.download_button("자율시간 승인 신청서 다운로드",
                           excel_data,
                           file_name=f"{st.session_state.data.get('activity_name','자율시간승인신청서')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    else:
        st.warning("최소 하나의 항목을 선택해주세요.")

    if st.button("다음 단계로", use_container_width=True):
        st.session_state.step = 3
        st.rerun()

def create_approval_excel_document(selected_fields):
    output = BytesIO()
    data = st.session_state.data
    all_fields = {
        "학교급": data.get('school_type',''),
        "대상 학년": ', '.join(data.get('grades',[])),
        "총 차시": data.get('total_hours',''),
        "운영 학기": ', '.join(data.get('semester',[])),
        "연계 교과": ', '.join(data.get('subjects',[])),
        "활동명": data.get('activity_name',''),
        "요구사항": data.get('requirements',''),
        "필요성": data.get('necessity',''),
        "개요": data.get('overview','')
    }
    selected_data = {k:v for k,v in all_fields.items() if k in selected_fields}
    df = pd.DataFrame({"항목": list(selected_data.keys()),
                       "내용": list(selected_data.values())})
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="자율시간 승인 신청서")
        ws = writer.sheets["자율시간 승인 신청서"]
        ws.set_column("A:A", 20)
        ws.set_column("B:B", 50)
    return output.getvalue()

def show_step_3(vector_store):
    st.markdown("<div class='step-header'><h3>3단계: 내용체계</h3></div>", unsafe_allow_html=True)

    if "generated_step_2" not in st.session_state:
        with st.form("generate_4sets"):
            st.info("영역명, 핵심 아이디어, 내용 요소를 **4세트** 생성합니다.")
            submit_btn = st.form_submit_button("4세트 생성 및 다음 단계로", use_container_width=True)
        if submit_btn:
            with st.spinner("생성 중..."):
                content = generate_content(3, st.session_state.data, vector_store)
                if isinstance(content,list) and len(content)==4:
                    st.session_state.data["content_sets"] = content
                    st.success("4세트 내용체계 생성 완료.")
                else:
                    st.warning("4세트 형태가 아닌 응답, 기본값 사용.")
                    st.session_state.data["content_sets"]=[]
                st.session_state.generated_step_2 = True
    else:
        content_sets = st.session_state.data.get("content_sets",[])
        with st.form("edit_4sets_form"):
            st.markdown("#### 생성된 4세트 내용체계 수정")
            new_sets = []
            tabs = st.tabs([f"내용체계 {i+1}" for i in range(4)])
            for i, tab in enumerate(tabs):
                with tab:
                    cset = content_sets[i] if i<len(content_sets) else {
                        "domain":"",
                        "key_ideas":[],
                        "content_elements":{
                            "knowledge_and_understanding":[],
                            "process_and_skills":[],
                            "values_and_attitudes":[]
                        }
                    }
                    domain_input = st.text_input("영역명", cset.get("domain",""), key=f"domain_{i}")
                    ki_list = cset.get("key_ideas",[])
                    ki_text = "\n".join(ki_list)
                    ki_input = st.text_area("핵심 아이디어", ki_text, height=80, key=f"ki_{i}")

                    ce = cset.get("content_elements",{})
                    kua = ce.get("knowledge_and_understanding",[])
                    pns = ce.get("process_and_skills",[])
                    vat = ce.get("values_and_attitudes",[])

                    col1,col2,col3 = st.columns(3)
                    with col1:
                        st.markdown("##### 지식·이해")
                        kua_text = "\n".join(kua)
                        kua_input = st.text_area("knowledge_and_understanding",
                                                 kua_text, height=120, key=f"kua_{i}")
                    with col2:
                        st.markdown("##### 과정·기능")
                        pns_text = "\n".join(pns)
                        pns_input = st.text_area("process_and_skills",
                                                 pns_text, height=120, key=f"pns_{i}")
                    with col3:
                        st.markdown("##### 가치·태도")
                        vat_text = "\n".join(vat)
                        vat_input = st.text_area("values_and_attitudes",
                                                 vat_text, height=120, key=f"vat_{i}")

                    new_sets.append({
                        "domain": domain_input,
                        "key_ideas": [line.strip() for line in ki_input.split("\n") if line.strip()],
                        "content_elements":{
                            "knowledge_and_understanding":[line.strip() for line in kua_input.split("\n") if line.strip()],
                            "process_and_skills":[line.strip() for line in pns_input.split("\n") if line.strip()],
                            "values_and_attitudes":[line.strip() for line in vat_input.split("\n") if line.strip()]
                        }
                    })
            save_btn = st.form_submit_button("4세트 저장 및 다음 단계로", use_container_width=True)

        if save_btn:
            with st.spinner("저장 중..."):
                st.session_state.data["content_sets"] = new_sets
                combined_kis = []
                for cset in new_sets:
                    combined_kis.extend(cset.get("key_ideas",[]))
                st.session_state.data["key_ideas"] = combined_kis

                if new_sets:
                    st.session_state.data["domain"] = new_sets[0]["domain"]
                    st.session_state.data["content_elements"] = new_sets[0]["content_elements"]
                else:
                    st.session_state.data["domain"] = ""
                    st.session_state.data["content_elements"] = {}

                del st.session_state.generated_step_2
                st.success("4세트 내용 저장 완료. 핵심 아이디어 반영.")
                st.session_state.step = 4
                st.rerun()

def show_step_4(vector_store):
    st.markdown("<div class='step-header'><h3>4단계: 성취기준 설정</h3></div>", unsafe_allow_html=True)
    content_sets = st.session_state.data.get("content_sets",[])
    num_sets = len(content_sets)

    if "generated_step_3" not in st.session_state:
        with st.form("standards_form"):
            st.info(f"내용체계 세트가 {num_sets}개이므로, 성취기준도 {num_sets}개를 생성합니다.")
            sub_btn = st.form_submit_button("생성 및 다음 단계로", use_container_width=True)
        if sub_btn:
            with st.spinner("성취기준 생성 중..."):
                standards = generate_content(4, st.session_state.data, vector_store)
                if isinstance(standards,list) and len(standards)==num_sets:
                    st.session_state.data["standards"] = standards
                    st.success(f"성취기준 {num_sets}개 생성 완료.")
                else:
                    st.warning(f"{num_sets}개 성취기준이 아님. 빈 리스트 저장.")
                    st.session_state.data["standards"] = []
                st.session_state.generated_step_3 = True
    else:
        with st.form("edit_standards_form"):
            st.markdown("#### 생성된 성취기준 수정")
            old_stds = st.session_state.data.get("standards",[])
            new_stds = []
            for i, std in enumerate(old_stds):
                st.markdown(f"##### 성취기준 {i+1}")
                code = st.text_input("성취기준 코드", std["code"], key=f"std_code_{i}")
                desc = st.text_area("성취기준 설명", std["description"], height=100, key=f"std_desc_{i}")

                st.markdown("##### 수준별 성취기준 (상, 중, 하)")
                lvls = std.get("levels",[])
                col1, col2, col3 = st.columns(3)
                with col1:
                    a_text = next((l["description"] for l in lvls if l["level"]=="A"),"")
                    a_in = st.text_area("상(A)", a_text, height=80, key=f"std_{i}_A")
                with col2:
                    b_text = next((l["description"] for l in lvls if l["level"]=="B"),"")
                    b_in = st.text_area("중(B)", b_text, height=80, key=f"std_{i}_B")
                with col3:
                    c_text = next((l["description"] for l in lvls if l["level"]=="C"),"")
                    c_in = st.text_area("하(C)", c_text, height=80, key=f"std_{i}_C")

                new_stds.append({
                    "code": code,
                    "description": desc,
                    "levels":[
                        {"level":"A","description":a_in},
                        {"level":"B","description":b_in},
                        {"level":"C","description":c_in}
                    ]
                })
                st.markdown("---")
            save_btn = st.form_submit_button("수정사항 저장 및 다음 단계로", use_container_width=True)
        if save_btn:
            with st.spinner("저장 중..."):
                st.session_state.data["standards"] = new_stds
                del st.session_state.generated_step_3
                st.success("성취기준 저장 완료.")
                st.session_state.step = 5
                st.rerun()

def show_step_5(vector_store):
    st.markdown("<div class='step-header'><h3>5단계: 교수학습 및 평가</h3></div>", unsafe_allow_html=True)

    if "generated_step_4" not in st.session_state:
        with st.form("teaching_assessment_form"):
            st.info("교수학습방법 및 평가계획을 자동으로 생성합니다.")
            sb = st.form_submit_button("생성 및 다음 단계로", use_container_width=True)
        if sb:
            with st.spinner("생성 중..."):
                result = generate_content(5, st.session_state.data, vector_store)
                if result:
                    st.session_state.data["teaching_methods_text"] = result.get("teaching_methods_text","")
                    st.session_state.data["assessment_plan"] = result.get("assessment_plan",[])
                    st.success("교수학습 및 평가 생성 완료.")
                else:
                    st.warning("교수학습 및 평가 생성 실패. 기본값 사용.")
                    st.session_state.data["teaching_methods_text"] = ""
                    st.session_state.data["assessment_plan"] = []
                st.session_state.generated_step_4 = True
    else:
        # 이미 평가계획 있음 -> 수정
        with st.form("edit_teaching_assessment_form"):
            st.markdown("#### 교수학습방법 (여러 줄 가능)")
            teaching_methods_text = st.text_area("교수학습방법",
                                                 value=st.session_state.data.get("teaching_methods_text",""),
                                                 height=120)
            st.markdown("#### 평가계획")
            old_plan = st.session_state.data.get("assessment_plan",[])
            new_plan = []

            for i, ap in enumerate(old_plan):
                code = ap.get("code","")
                desc = ap.get("description","")
                elem = ap.get("element","")
                meth = ap.get("method","")
                ch = ap.get("criteria_high","")
                cm = ap.get("criteria_mid","")
                cl = ap.get("criteria_low","")

                st.markdown(f"##### 평가항목 {i+1}")
                row1_col1, row1_col2, row1_col3 = st.columns([2,2,2])
                with row1_col1:
                    st.markdown(f"**코드**: `{code}`")
                    st.markdown(f"**성취기준**: {desc}")
                with row1_col2:
                    new_elem = st.text_area("평가요소", elem, height=80, key=f"elem_{code}")
                with row1_col3:
                    new_meth = st.text_area("수업평가방법", meth, height=80, key=f"meth_{code}")

                st.markdown("**평가기준(상·중·하)**")
                high_in = st.text_area("상(A)", ch, height=80, key=f"critH_{code}")
                mid_in = st.text_area("중(B)", cm, height=80, key=f"critM_{code}")
                low_in = st.text_area("하(C)", cl, height=80, key=f"critL_{code}")

                new_plan.append({
                    "code": code,
                    "description": desc,
                    "element": new_elem,
                    "method": new_meth,
                    "criteria_high": high_in,
                    "criteria_mid": mid_in,
                    "criteria_low": low_in
                })
                st.markdown("---")
            sb2 = st.form_submit_button("수정사항 저장 및 다음 단계로", use_container_width=True)

        if sb2:
            with st.spinner("수정사항 저장 중..."):
                st.session_state.data["teaching_methods_text"] = teaching_methods_text
                st.session_state.data["assessment_plan"] = new_plan
                del st.session_state.generated_step_4
                st.success("교수학습 및 평가 수정 완료.")
                st.session_state.step = 6
                st.rerun()


def generate_all_lesson_plans(total_hours, data, vector_store=None, query_for_retrieval="학교자율시간 차시별 지도계획"):
    """
    한 번에 모든 차시를 생성하되, vector_store가 있을 경우 'query_for_retrieval'로
    관련 문서만 검색하여 컨텍스트로 활용하는 함수.
    """
    from langchain.schema import SystemMessage, HumanMessage
    from langchain_openai import ChatOpenAI

    # (1) 필요한 데이터
    domain           = data.get('domain','')
    key_ideas        = data.get('key_ideas',[])
    content_elements = data.get('content_elements',{})
    standards        = data.get('standards',[])
    teaching_methods = data.get('teaching_methods_text','')
    assessment_plan  = data.get('assessment_plan',[])
    activity_name    = data.get('activity_name','')
    requirements     = data.get('requirements','')
    openai_api_key   = OPENAI_API_KEY  # 혹은 data.get("OPENAI_API_KEY", "")

    # (2) 벡터스토어: 관련성 있는 문서만 retrieval
    doc_context = ""
    if vector_store is not None:
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k":5})
            # query_for_retrieval로 검색
            results = retriever.get_relevant_documents(query_for_retrieval)
            doc_context = "\n\n".join([doc.page_content for doc in results])
        except Exception as e:
            print("[WARN] 벡터스토어 검색 오류:", e)
            doc_context = ""

    # (3) 프롬프트
    SYSTEM = """한국의 초등학교 2022 개정 교육과정 전문가입니다.
    학교자율시간 계획서를 다음 원칙에 따라 작성합니다:

    1. 지도계획에 모든 차시에 학습내용과 학습 주제가 빈틈없이 내용이 꼭 들어가야 합니다.
    2. 학습자 중심의 교육과정 구성
    3. 실생활 연계 및 체험 중심 활동
    4. 교과 간 연계 및 통합적 접근
    5. 과정 중심 평가와 피드백 강조
    6. 유의미한 학습경험 제공
    7. 요구사항을 반영한 맞춤형 교육과정 구성
    8. 교수학습 방법의 다양화
    9. 객관적이고 공정한 평가계획 수립
    """

    user_prompt = f"""
다음 정보를 종합하여 1차시부터 {total_hours}차시까지 모두 연결된 흐름으로 '차시별 지도계획'을 JSON으로 작성해주세요.

[이전 단계 결과]
- 영역명: {domain}
- 핵심 아이디어: {key_ideas}
- 내용체계: {content_elements}
- 성취기준: {standards}
- 교수학습방법: {teaching_methods}
- 평가계획: {assessment_plan}
- 활동명: {activity_name}
- 요구사항: {requirements}

추가 문서(관련성 있는 자료만):
{doc_context}

지침:
1. 아래 예시를 참고하여 작성해주세요.
(예시)
학습주제: 질문 약속 만들기
학습내용: 질문을 할 때 지켜야 할 약속 만들기
         수업 중 질문, 일상 속 질문 속에서 갖추어야 할 예절 알기
2. 각 차시에 대해 lesson_number, topic(학습주제), content(학습내용), materials(교수학습자료) 필수
3. 1차시부터 {total_hours}차시까지 빠짐없이 작성
4. 이전 차시와의 연계를 고려 (중복X, 연계O)
5. JSON 형식만 반환(예시):
{{
  "lesson_plans": [
    {{
      "lesson_number": "1",
      "topic": "차시별 학습주제",
      "content": "학습내용",
      "materials": "교수학습자료"
    }},
    ...
  ]
}}
"""

    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=user_prompt)
    ]

    chat = ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-4o",
        temperature=0.7,
        max_tokens=3000
    )

    try:
        resp = chat(messages)
        raw_json = resp.content.strip()
        raw_json = raw_json.replace("```json","").replace("```","").strip()
        parsed = json.loads(raw_json)
        lesson_plans = parsed.get("lesson_plans", [])
        # 혹시 번호가 불규칙하면 보정
        for i, plan in enumerate(lesson_plans, start=1):
            plan["lesson_number"] = str(i)
        return lesson_plans
    except Exception as e:
        print("[ERROR] generate_all_lesson_plans 오류:", e)
        return []


def show_step_6(vector_store):
    total_hours = st.session_state.data.get('total_hours',29)
    st.markdown(f"<div class='step-header'><h3>6단계: 차시별 지도계획 ({total_hours}차시)</h3></div>", unsafe_allow_html=True)

    if "generated_step_5" not in st.session_state:
        with st.form("lesson_plans_form"):
            st.info(f"{total_hours}차시 계획을 한 번에 자동 생성합니다.")
            sb = st.form_submit_button(f"{total_hours}차시 생성 및 다음 단계로", use_container_width=True)
        if sb:
            with st.spinner("전체 차시별 계획 생성 중..."):
                # 관련성 있는 자료만 검색하고 싶으면 특정 쿼리를 전달
                query_for_retrieval = "학교자율시간 차시별 지도계획"
                all_plans = generate_all_lesson_plans(
                    total_hours,
                    st.session_state.data,
                    vector_store=vector_store,
                    query_for_retrieval=query_for_retrieval
                )
                if all_plans:
                    st.session_state.data["lesson_plans"] = all_plans
                    st.success(f"{total_hours}차시 계획 생성 완료.")
                    st.session_state.generated_step_5 = True
                else:
                    st.warning("차시별 계획 생성 실패. 빈 리스트로 저장합니다.")
                    st.session_state.data["lesson_plans"] = []
    else:
        # 이미 생성된 경우 -> 수정 모드
        with st.form("edit_lesson_plans_form"):
            st.markdown("#### 생성된 차시별 계획 수정")
            lesson_plans = st.session_state.data.get("lesson_plans",[])
            edited_plans = []

            # 한 탭에 전부 보여줄 수도 있고, 10차시씩 탭을 나눠 보여줄 수도 있음
            total_tabs = (total_hours+9)//10
            tabs = st.tabs([f"{i*10+1}~{min((i+1)*10,total_hours)}차시" for i in range(total_tabs)])
            for tab_idx, tab in enumerate(tabs):
                with tab:
                    start_idx = tab_idx*10
                    end_idx = min(start_idx+10, total_hours)
                    for i in range(start_idx, end_idx):
                        st.markdown(f"##### {i+1}차시")
                        col1,col2 = st.columns([1,2])
                        with col1:
                            topic = st.text_input("학습주제", lesson_plans[i].get("topic",""), key=f"topic_{i}")
                            materials = st.text_input("교수학습자료", lesson_plans[i].get("materials",""), key=f"materials_{i}")
                        with col2:
                            content = st.text_area("학습내용", lesson_plans[i].get("content",""), height=100, key=f"content_{i}")
                        edited_plans.append({
                            "lesson_number": f"{i+1}",
                            "topic": topic,
                            "content": content,
                            "materials": materials
                        })
                        st.markdown("---")

            sb2 = st.form_submit_button("수정사항 저장 및 다음 단계로", use_container_width=True)

        if sb2:
            with st.spinner("저장 중..."):
                st.session_state.data["lesson_plans"] = edited_plans
                del st.session_state.generated_step_5
                st.success("차시별 계획 수정 완료.")
                st.session_state.step = 7
                st.rerun()

def show_final_review(vector_store):
    st.title("최종 계획서 검토")
    try:
        data = st.session_state.data
        tabs = st.tabs(["기본정보", "내용체계", "성취기준", "교수학습 및 평가", "차시별계획"])

        # 1) 기본정보
        with tabs[0]:
            st.markdown("### 기본 정보")
            basic_info = {
                "학교급": data.get('school_type',''),
                "대상 학년": ', '.join(data.get('grades',[])),
                "총 차시": f"{data.get('total_hours','')}차시",
                "주당 차시": f"{data.get('weekly_hours','')}차시",
                "운영 학기": ', '.join(data.get('semester',[])),
                "연계 교과": ', '.join(data.get('subjects',[])),
                "활동명": data.get('activity_name',''),
                "요구사항": data.get('requirements',''),
                "필요성": data.get('necessity',''),
                "개요": data.get('overview','')
            }
            for k,v in basic_info.items():
                st.markdown(f"**{k}**: {v}")

            st.button("기본정보 수정하기", key="edit_basic_info",
                      on_click=lambda: set_step(1),
                      use_container_width=True)

        # 2) 내용체계
        with tabs[1]:
            st.markdown("### 내용체계 (4세트)")
            content_sets = data.get("content_sets",[])
            if not content_sets:
                st.warning("현재 저장된 내용체계가 없습니다.")
            else:
                for i, cset in enumerate(content_sets,start=1):
                    st.markdown(f"#### ▶ 내용체계 세트 {i}")
                    domain = cset.get("domain","")
                    key_ideas = cset.get("key_ideas",[])
                    content_elements = cset.get("content_elements",{})

                    st.write(f"**영역명**: {domain}")
                    st.write("**핵심 아이디어**:")
                    if key_ideas:
                        for idea in key_ideas:
                            st.write(f"- {idea}")
                    else:
                        st.write("- (없음)")

                    st.write("**내용 요소**:")
                    kua = content_elements.get("knowledge_and_understanding",[])
                    pns = content_elements.get("process_and_skills",[])
                    vat = content_elements.get("values_and_attitudes",[])

                    st.markdown("- 지식·이해")
                    for item in kua:
                        st.write(f"  - {item}")
                    st.markdown("- 과정·기능")
                    for item in pns:
                        st.write(f"  - {item}")
                    st.markdown("- 가치·태도")
                    for item in vat:
                        st.write(f"  - {item}")

                    st.divider()

            st.button("내용체계 수정하기", key="edit_content_sets",
                      on_click=lambda: set_step(3),
                      use_container_width=True)

        # 3) 성취기준
        with tabs[2]:
            st.markdown("### 성취기준")
            for std in data.get("standards",[]):
                st.markdown(f"**{std['code']}**: {std['description']}")
                st.markdown("##### 수준별 성취기준")
                for lv in std["levels"]:
                    label_map = {"A":"상","B":"중","C":"하"}
                    label = label_map.get(lv["level"], lv["level"])
                    st.write(f"- {label} 수준: {lv['description']}")
                st.markdown("---")

            st.button("성취기준 수정하기",
                      key="edit_standards",
                      on_click=lambda: set_step(4),
                      use_container_width=True)

        # 4) 교수학습 및 평가
        with tabs[3]:
            st.markdown("### 교수학습 및 평가")
            methods_text = data.get("teaching_methods_text","")
            st.markdown("#### 교수학습방법")
            if methods_text.strip():
                lines = methods_text.split('\n')
                for line in lines:
                    st.write(f"- {line.strip()}")
            else:
                st.write("(교수학습방법 없음)")

            st.markdown("#### 평가계획")
            for ap in data.get("assessment_plan",[]):
                code = ap.get("code","")
                desc = ap.get("description","")
                elem = ap.get("element","")
                meth = ap.get("method","")
                hi = ap.get("criteria_high","")
                mi = ap.get("criteria_mid","")
                lo = ap.get("criteria_low","")

                st.markdown(f"**{code}** - {desc}")
                st.write(f"- 평가요소: {elem}")
                st.write(f"- 수업평가방법: {meth}")
                st.write(f"- 상 수준 기준: {hi}")
                st.write(f"- 중 수준 기준: {mi}")
                st.write(f"- 하 수준 기준: {lo}")
                st.markdown("---")

            st.button("교수학습 및 평가 수정하기",
                      key="edit_teaching_assessment",
                      on_click=lambda: set_step(5),
                      use_container_width=True)

        # 5) 차시별계획
        with tabs[4]:
            st.markdown("### 차시별 계획")
            lesson_plans_df = pd.DataFrame(data.get("lesson_plans",[]))
            if not lesson_plans_df.empty:
                # 표시용 컬럼 이름 매핑
                col_map = {
                    "lesson_number":"차시",
                    "topic":"학습주제",
                    "content":"학습내용",
                    "materials":"교수학습자료"
                }
                st.dataframe(lesson_plans_df,
                             column_config=col_map,
                             hide_index=True,
                             height=400)
            else:
                st.warning("차시별 계획이 없습니다.")

            st.button("차시별 계획 수정하기",
                      key="edit_lesson_plans",
                      on_click=lambda: set_step(6),
                      use_container_width=True)

        # 하단 버튼들
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("모든 단계 수정하기", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        with col2:
            st.markdown("#### 원하는 항목만 선택하여 Excel 다운로드")
            available_sheets = ["기본정보","내용체계","성취기준","교수학습 및 평가","차시별계획"]
            selected_sheets = st.multiselect("다운로드할 항목",
                                             options=available_sheets,
                                             default=available_sheets)
            if selected_sheets:
                excel_data = create_excel_document(selected_sheets)
                st.download_button("📥 Excel 다운로드",
                                   excel_data,
                                   file_name=f"{data.get('activity_name','학교자율시간계획서')}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
            else:
                st.warning("최소 한 개 이상의 항목을 선택해주세요.")
        with col3:
            if st.button("새로 만들기", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    except Exception as e:
        st.error(f"최종 검토 처리 중 오류: {e}")

def create_excel_document(selected_sheets):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        content_format = workbook.add_format({
            'text_wrap':True,
            'valign':'top',
            'border':1
        })
        data = st.session_state.data

        if "기본정보" in selected_sheets:
            df = pd.DataFrame([{
                '학교급': data.get('school_type',''),
                '대상학년': ', '.join(data.get('grades',[])),
                '총차시': data.get('total_hours',''),
                '주당차시': data.get('weekly_hours',''),
                '운영 학기': ', '.join(data.get('semester',[])),
                '연계 교과': ', '.join(data.get('subjects',[])),
                '활동명': data.get('activity_name',''),
                '요구사항': data.get('requirements',''),
                '필요성': data.get('necessity',''),
                '개요': data.get('overview','')
            }])
            df.T.to_excel(writer, sheet_name='기본정보', header=['내용'])
            ws = writer.sheets['기본정보']
            ws.set_column(0,0,30, content_format)
            ws.set_column(1,1,80, content_format)

        if "내용체계" in selected_sheets:
            content_sets = data.get("content_sets",[])
            if not content_sets:
                df_empty = pd.DataFrame([{"구분":"내용체계 없음","내용":""}])
                df_empty.to_excel(writer, sheet_name='내용체계', index=False)
                ws = writer.sheets['내용체계']
                ws.set_column('A:A',20, content_format)
                ws.set_column('B:B',80, content_format)
            else:
                rows=[]
                for i,cset in enumerate(content_sets, start=1):
                    domain=cset.get("domain","")
                    key_ideas=cset.get("key_ideas",[])
                    ce=cset.get("content_elements",{})

                    rows.append({"구분":f"영역명(세트{i})","내용":domain})
                    for ki in key_ideas:
                        rows.append({"구분":f"핵심 아이디어(세트{i})","내용":ki})
                    for k in ce.get("knowledge_and_understanding",[]):
                        rows.append({"구분":f"지식·이해(세트{i})","내용":k})
                    for p in ce.get("process_and_skills",[]):
                        rows.append({"구분":f"과정·기능(세트{i})","내용":p})
                    for v in ce.get("values_and_attitudes",[]):
                        rows.append({"구분":f"가치·태도(세트{i})","내용":v})

                dfc = pd.DataFrame(rows)
                dfc.to_excel(writer, sheet_name='내용체계', index=False)
                ws = writer.sheets['내용체계']
                ws.set_column('A:A',30, content_format)
                ws.set_column('B:B',80, content_format)

        if "성취기준" in selected_sheets:
            standards = data.get("standards",[])
            st_rows=[]
            for std in standards:
                code = std.get("code","")
                desc = std.get("description","")
                lv = std.get("levels",[])
                for level in lv:
                    lv_code = level.get("level","?")
                    lv_desc = level.get("description","")
                    label_map={"A":"상","B":"중","C":"하"}
                    st_rows.append({
                        "성취기준코드":code,
                        "성취기준설명":desc,
                        "수준":label_map.get(lv_code, lv_code),
                        "수준별설명":lv_desc
                    })
            dfstd = pd.DataFrame(st_rows)
            dfstd.to_excel(writer, sheet_name='성취기준', index=False)
            ws = writer.sheets['성취기준']
            ws.set_column('A:A',15, content_format)
            ws.set_column('B:B',50, content_format)
            ws.set_column('C:C',10, content_format)
            ws.set_column('D:D',80, content_format)

        if "교수학습 및 평가" in selected_sheets:
            rows=[]
            teaching_text=data.get("teaching_methods_text","").strip()
            if teaching_text:
                lines=teaching_text.split("\n")
                for line in lines:
                    if line.strip():
                        rows.append({
                            "유형":"교수학습방법",
                            "코드":"",
                            "성취기준":"",
                            "평가요소":"",
                            "수업평가방법":line.strip(),
                            "상":"",
                            "중":"",
                            "하":""
                        })
            ap=data.get("assessment_plan",[])
            for a in ap:
                rows.append({
                    "유형":"평가계획",
                    "코드":a.get("code",""),
                    "성취기준":a.get("description",""),
                    "평가요소":a.get("element",""),
                    "수업평가방법":a.get("method",""),
                    "상":a.get("criteria_high",""),
                    "중":a.get("criteria_mid",""),
                    "하":a.get("criteria_low","")
                })
            dfm = pd.DataFrame(rows)
            dfm.to_excel(writer, sheet_name='교수학습및평가', index=False)
            ws = writer.sheets['교수학습및평가']
            ws.set_column('A:A',14, content_format)
            ws.set_column('B:B',14, content_format)
            ws.set_column('C:C',30, content_format)
            ws.set_column('D:D',30, content_format)
            ws.set_column('E:E',30, content_format)
            ws.set_column('F:F',30, content_format)
            ws.set_column('G:G',30, content_format)
            ws.set_column('H:H',30, content_format)

        if "차시별계획" in selected_sheets:
            lp = data.get("lesson_plans",[])
            if lp:
                df_lp = pd.DataFrame(lp)
                df_lp.columns=["차시","학습주제","학습내용","교수학습자료"]
                df_lp.to_excel(writer, sheet_name='차시별계획', index=False)
                ws = writer.sheets['차시별계획']
                ws.set_column('A:A',10,content_format)
                ws.set_column('B:B',30,content_format)
                ws.set_column('C:C',80,content_format)
                ws.set_column('D:D',50,content_format)
            else:
                df_empty = pd.DataFrame([{"차시":"","학습주제":"","학습내용":"","교수학습자료":""}])
                df_empty.to_excel(writer, sheet_name='차시별계획', index=False)
                ws = writer.sheets['차시별계획']
                ws.set_column('A:A',10,content_format)
                ws.set_column('B:B',30,content_format)
                ws.set_column('C:C',80,content_format)
                ws.set_column('D:D',50,content_format)

    return output.getvalue()

def set_step(step_number):
    st.session_state.step = step_number


def show_chatbot(global_vs):
    st.sidebar.markdown("## 학교자율시간 교육과정 설계 챗봇")

    st.sidebar.markdown("**추천 질문:**")
    recommended_questions = [
        "초등학교 3학년 학교자율시간의 활동 10가지만 제시하여 주세요?",
        "자율시간 운영에 필요한 자료는 무엇인가요?",
        "자율시간 수업의 효과적인 진행 방법은?"
    ]
    with st.sidebar.container():
        st.markdown('<div class="sidebar-questions">', unsafe_allow_html=True)
        for q in recommended_questions:
            if st.sidebar.button(q, key=f"rec_{q}"):
                st.session_state.chat_input = q
        st.markdown('</div>', unsafe_allow_html=True)

    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""
    user_input = st.sidebar.text_input("질문을 입력하세요:", value=st.session_state.chat_input, key="chat_input")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.sidebar.button("질문 전송", key="send_question"):
        if user_input:
            if not global_vs:
                st.sidebar.warning("챗봇 인덱스가 없습니다.")
            else:
                # 전체 문서 검색 (k=5 등 적절히)
                retriever = global_vs.as_retriever(search_kwargs={"k":5})
                results = retriever.get_relevant_documents(user_input)
                context = "\n\n".join([doc.page_content for doc in results])

                prompt = f"""당신은 귀여운 친구 캐릭터 두 명, '🐰 토끼'와 '🐻 곰돌이'입니다.
두 캐릭터는 협력하여 학교자율시간 관련 질문에 대해 번갈아 가며 귀엽고 친근한 말투로 답변합니다.
- 문서와 모순되는 내용은 쓰지 않기
- 문서에 없는 내용은 최소화
- 문서 내용이 있으면 활용
질문: {user_input}
관련 정보: {context}
답변:"""
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt)
                ]
                chat = ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    model="gpt-4",
                    temperature=0.7,
                    max_tokens=512
                )
                response = chat(messages)
                answer = response.content.strip()

                st.sidebar.markdown("**🤖 답변:**")
                sidebar_typewriter_effect(answer, delay=0.001)
                st.session_state.chat_history.append((user_input, answer))
        else:
            st.sidebar.warning("질문을 입력해주세요.")

    if st.session_state.chat_history:
        st.sidebar.markdown("### 대화 내역")
        for idx, (q, a) in enumerate(st.session_state.chat_history):
            st.sidebar.markdown(f"**Q{idx+1}:** {q}")
            st.sidebar.markdown(f"**🤖 A{idx+1}:** {a}")


def main():
    try:
        set_page_config()

        if "data" not in st.session_state:
            st.session_state.data = {}
        if "step" not in st.session_state:
            st.session_state.step = 1

        st.title("2022 개정 교육과정 학교자율시간 계획서 생성기")

        # (1) 챗봇용 전체 문서 인덱스
        global_vector_store = setup_combined_vector_store()

        # (2) 진행바
        show_progress()

        step_funcs = {
            1: show_step_1,
            2: show_step_2_approval,
            3: show_step_3,
            4: show_step_4,
            5: show_step_5,
            6: show_step_6,
            7: show_final_review
        }
        current_step = st.session_state.step
        func = step_funcs.get(current_step)

        if func:
            # 단계별 문서 인덱스 로딩
            step_vector_store = get_vector_store_for_step(current_step)
            # 해당 단계 UI
            func(step_vector_store)
        else:
            st.error("잘못된 단계입니다.")

        # 사이드바 챗봇 (전체 문서 인덱스)
        show_chatbot(global_vector_store)

    except Exception as e:
        st.error(f"애플리케이션 실행 중 오류: {e}")
        if st.button("처음부터 다시 시작"):
            st.session_state.clear()
            st.rerun()

if __name__=="__main__":
    main()
