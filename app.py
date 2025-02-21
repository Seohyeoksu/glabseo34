import os
import streamlit as st
import pandas as pd
from io import BytesIO
import json
import time

from langchain.prompts import ChatPromptTemplate
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.document_loaders import UnstructuredPDFLoader



OPENAI_API_KEY = st.secrets["openai"]["api_key"]
if not OPENAI_API_KEY:
    st.error("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 확인하세요.")
    st.stop()

SYSTEM_PROMPT = """한국의 초등학교 2022 개정 교육과정 전문가입니다.
학교자율시간 계획서를 다음 원칙에 따라 작성합니다:

1. 지도계획에 모든 차시에 학습내용과 학습 주제가 빈틈없이 내용이 꼭 들어가야 합니다.
2. 학습자 중심의 교육과정 초등학교 3,4학년 수준에 맞는 쉽게 내용을 만들어 주세요.
3. 실생활 연계 및 체험 중심 활동
4. 교과 간 연계 및 통합적 접근
5. 초등학교 3학년, 4학년 수준에 맞아야 한다. 
7. 요구사항을 반영한 맞춤형 교육과정 구성
8. 교수학습 방법의 다양화
9. 객관적이고 공정한 평가계획 수립
10.초등학교 수준에 맞는 내용 구성성
"""


def sidebar_typewriter_effect(text, delay=0.001):
    placeholder = st.sidebar.empty()
    output = ""
    for char in text:
        output += char
        placeholder.markdown(output)
        time.sleep(delay)
    return output




def set_page_config():
    try:
        st.set_page_config(page_title="학교자율시간 올인원", page_icon="📚", layout="wide")
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



@st.cache_resource(show_spinner="문서 로딩 완료...")
def setup_vector_store():
    try:
        index_dir = "faiss_index"
        if os.path.exists(index_dir) and os.path.isdir(index_dir):
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vector_store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            return vector_store
        else:
            st.info("기존 인덱스가 없어, 문서를 로드 후 임베딩합니다. (처음 한 번만 실행)")

            documents_dir = "./documents/"
            supported_extensions = ["pdf", "txt", "docx"]
            all_docs = []

            for filename in os.listdir(documents_dir):
                if any(filename.lower().endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(documents_dir, filename)
                    if filename.lower().endswith(".pdf"):
                        loader = UnstructuredPDFLoader(file_path)
                    else:
                        loader = UnstructuredLoader(file_path)
                    documents = loader.load()
                    all_docs.extend(documents)

            if not all_docs:
                st.error("documents/ 폴더에 문서가 없습니다.")
                return None

            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vector_store = FAISS.from_documents(all_docs, embeddings)
            vector_store.save_local(index_dir)
            st.success("새로운 벡터 스토어가 생성되어 저장되었습니다.")
            return vector_store

    except Exception as e:
        st.error(f"벡터 스토어 설정 중 오류: {str(e)}")
        return None



def generate_content(step, data, vector_store):
    """
    step별로 AI 프롬프트를 구성하고 JSON 형식의 응답을 받아 parsing하는 함수
    """
    # (1) step에 따른 검색 키워드(쿼리) 설정
    query_map = {
        3: "내용체계",
        4: "성취기준",
        5: "교수학습 및 평가"
    }

    try:
        # (2) 검색 컨텍스트 준비
        context = ""
        if step >= 3 and vector_store:
            retriever = vector_store.as_retriever()
            queries = query_map.get(step, "")
            context_docs = []

            
            if isinstance(queries, list):
                for q in queries:
                    docs = retriever.get_relevant_documents(q)
                    context_docs.extend(docs)
            else:
               
                docs = retriever.get_relevant_documents(queries)
                context_docs.extend(docs)

         
            unique_dict = {}
            for d in context_docs:
                unique_dict[d.page_content] = d
            unique_docs = list(unique_dict.values())

           
            context = "\n\n".join(doc.page_content for doc in unique_docs)

       

        necessity = data.get('necessity', '')
        overview = data.get('overview', '')
        standards = data.get('standards', [])
        content_sets = data.get("content_sets", [])
        num_sets = len(content_sets)

       
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

            # 3단계: 내용체계
            3: f"""
            
{context}
문서를 그대로 가져오는 것은 안되고 활동명: {data.get('activity_name')} 부합되도록 사용해야 한다.
활동명: {data.get('activity_name')}
요구사항: {data.get('requirements')}을 가장 많이 반영해서 작성하면면 좋겠어.
학교급: {data.get('school_type')}도 반영해야 한다. 
대상 학년: {', '.join(data.get('grades', []))}을 고려해서 작성해야 한다.
연계 교과: {', '.join(data.get('subjects', []))}
이전 단계 결과를 참고하여 작성하기
 핵심 아이디어는 IB교육육에서 이야기 하는 빅아이디어와 같은 거야. 학생들이 도달 할 수 있는 일반화된 이론이야 예시처럼 문장으로 진술해주세요.
'영역명(domain)', '핵심 아이디어(key_ideas)', '내용 요소(content_elements)'(지식·이해 / 과정·기능 / 가치·태도) 4개 세트를 생성... 를 JSON 구조로 작성해주세요. 
'content_elements'에는 **'knowledge_and_understanding'(지식·이해), 'process_and_skills'(과정·기능), 'values_and_attitudes'(가치·태도)**가 반드시 포함되어야 합니다.
예시를 참고하여 작성해주세요.
영역명도 창의적으로 다르게 구성하여 주세요 
<예시>
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
JSON 형식으로만 작성하고, 불필요한 문장은 쓰지 마세요.”, “추가 문장 없이 JSON만 반환
총 4개의 객체가 있는 JSON 배열

JSON 예시:
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

            # 4단계: 성취기준
            4: f"""{context}
이전 단계계
활동명: {data.get('activity_name')}
요구사항: {data.get('requirements')}
학교급: {data.get('school_type')}
대상 학년: {', '.join(data.get('grades', []))}
연계 교과: {', '.join(data.get('subjects', []))} 
내용 체계: {content_sets}

총 {num_sets}개 내용체계 세트가 생성되었으므로, 성취기준도 {num_sets}개 생성.

아래는 학년/교과/활동명에서 추출한 코드 접두사입니다:
code_prefix: "{make_code_prefix(data.get('grades', []), data.get('subjects', []), data.get('activity_name',''))}"

지침:
1. 성취기준코드는 반드시 code_prefix에 -01, -02, ... 식으로 순서 붙여 생성.
2. 성취기준은 내용체계표와 내용이 비슷하고 문장의 형식은 아래 예시를 참고:
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

            # 5단계: 교수학습 및 평가 → 상/중/하를 각각 별도 필드로 생성
            5: f"""{context}
이전 단계(성취기준): {standards}
1.평가요소, 수업평가방법, 평가기준은 예시문을 참고해서 작성해주세요
2.평가기준은 상,중,하로 나누어서 작성하여 주세요.
3.평가요소는 ~하기 형식으로 만들어 주세요.
4.다시 강조하지만 예시문 아래 예시문 형식으로 작성하여 주세요
<예시>
평가요소
 • 국가유산의 의미와 유형 알아보고 가치 탐색하기
수업평가방법
 [개념학습/프로젝트]
 • 국가유산의 의미를 이해하게 한 후 기준을 세워 국가유산을 유형별로 알아보고 문화유산의 가치를 파악하는지 평가하기
평가기준
 • 상:국가유산의 의미와 유형을 정확하게 이해하고 지역의 국가유산 조사를 통해 국가유산의 가치를 설명할 수 있다.
 • 중:국가유산의 의미와 유형을 이해하고 지역의 국가유산 조사를 통해 국가유산의 가치를 설명할 수 있다.
 • 하:주변의 도움을 받아 국가유산의 의미와 유형을 설명할 수 있다.

"teaching_methods_text"교수학습도 예시문을 참고해서 작성하여 주세요
<예시>
• 인간 활동으로 발생한 환경 영향의 긍정적인 사례와 부정적인 사례를 균형적으로 탐구하여 인간과 환경에 대한 다양한 측면을 이해하도록 한다.
• 다양한 사례를 통하여 환경오염의 현상을 이해하도록 지도하고 지속가능한 발전으로 이어질 수 있도록 내면화에 노력한다. 
• 학교나 지역의 다양한 체험활동 장소와 주제에 따른 계절을 고려하여 학습계획을 세워 학습을 진행한다. 
• 탐구 및 활동 시에는 사전 준비와 안전교육 등을 통하여 탐구과정에서 발생할 수 있는 안전사고를 예방하도록 한다. 
"teaching_methods_text": 문자열 (여러 줄 가능),
"assessment_plan": 리스트, 각 항목 =
  code (4단계 성취기준코드, 수정불가),
  description (4단계 성취기준문장, 수정불가),
  element (평가요소),
  method (수업평가방법),
  criteria (평가기준).
아래 예시 형식으로 JSON을 작성해주세요.
- 평가기준은 '상', '중', '하' 각각을 별도 필드로 기재 (criteria_high, criteria_mid, criteria_low)

JSON 예시:
{{
  "teaching_methods_text": "교수학습방법 여러 줄...",
  "assessment_plan": [
    {{
      "code": "성취기준코드(예: code_prefix-01)",
      "description": "성취기준문장",
      "element": "평가요소",
      "method": "수업평가방법",
      "criteria_high": "상 수준 평가기준",
      "criteria_mid": "중 수준 평가기준",
      "criteria_low": "하 수준 평가기준"
    }},
    ...
  ]
}}
"""
        }

        # step 2, 6, 7은 별도의 프롬프트 없이 빈 dict 반환
        if step in [2, 6, 7]:
            return {}

        prompt = step_prompts.get(step, "")
        if not prompt:
            return {}

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
        raw_text = response.content.strip().replace('```json','').replace('```','').strip()

        try:
            parsed = json.loads(raw_text)
            # 5단계 검증 → criteria_high/mid/low 세 개 모두 있는지 검사
            if step == 5:
                if not isinstance(parsed, dict):
                    raise ValueError("5단계 응답은 dict여야 합니다.")
                if "teaching_methods_text" not in parsed or "assessment_plan" not in parsed:
                    raise ValueError("teaching_methods_text, assessment_plan 키가 모두 필요.")
                for ap in parsed["assessment_plan"]:
                    for field in ["code","description","element","method","criteria_high","criteria_mid","criteria_low"]:
                        if field not in ap:
                            raise ValueError(f"assessment_plan 항목에 '{field}' 누락")
            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            st.warning(f"JSON 파싱 오류(단계 {step}): {e} → 빈 dict 반환")
            return {}

    except Exception as exc:
        st.error(f"generate_content({step}) 실행 중 오류: {exc}")
        return {}




def show_step_1(vector_store):
    st.markdown("<div class='step-header'><h3>1단계: 기본 정보</h3></div>", unsafe_allow_html=True)

    # school_type, grades, subjects 기본값 설정
    if "school_type" not in st.session_state.data:
        st.session_state.data["school_type"] = "초등학교"
    if "grades" not in st.session_state.data:
        st.session_state.data["grades"] = []
    if "subjects" not in st.session_state.data:
        st.session_state.data["subjects"] = []

    current_school_type = st.session_state.data.get('school_type', '초등학교')

 
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
            index = 0 if st.session_state.data["school_type"] == "초등학교" else 1
            school_type = st.radio("학교급", options, index=index)

            total_hours = st.number_input(
                "총 차시",
                min_value=1, max_value=68,
                value=st.session_state.data.get('total_hours', 34),
                help="총 차시 입력"
            )

            semester = st.multiselect(
                "운영 학기",
                ["1학기", "2학기"],
                default=st.session_state.data.get('semester', ["1학기"])
            )

            st.markdown("#### 학년 선택")
            if school_type == "초등학교":
                grades = st.multiselect(
                    "학년",
                    ["3학년", "4학년", "5학년", "6학년"],
                    default=st.session_state.data.get('grades', [])
                )
                subjects = st.multiselect(
                    "교과",
                    ["국어", "수학", "사회", "과학", "영어", "음악", "미술", "체육", "실과", "도덕"],
                    default=st.session_state.data.get('subjects', [])
                )
            else:
                grades = st.multiselect(
                    "학년",
                    ["1학년", "2학년", "3학년"],
                    default=st.session_state.data.get('grades', [])
                )
                subjects = st.multiselect(
                    "교과",
                    ["국어", "수학", "사회/역사", "과학/기술", "영어", "음악", "미술", "체육", "정보", "도덕","보건","진로와 직업","한문","환경과 녹생성장"],
                    default=st.session_state.data.get('subjects', [])
                )

            activity_name = st.text_input(
                "활동명",
                value=st.session_state.data.get('activity_name', ''),
                placeholder="예: 인공지능 놀이터"
            )
            requirements = st.text_area(
                "요구사항",
                value=st.session_state.data.get('requirements', ''),
                placeholder="예) 디지털 리터러시 강화 필요\n예) 학생들의 주도적 학습활동 및 안전교육 병행\n등등...",
                help="필요한 요구사항이나 핵심 요구 내용을 적어주세요.",
                height=100
            )

            submit_button = st.form_submit_button("정보 생성 및 다음 단계로", use_container_width=True)

        if submit_button:
            if activity_name and requirements and grades and subjects and semester:
                with st.spinner("정보 생성 중..."):
                    st.session_state.data["school_type"] = school_type
                    st.session_state.data["grades"] = grades
                    st.session_state.data["subjects"] = subjects
                    st.session_state.data["activity_name"] = activity_name
                    st.session_state.data["requirements"] = requirements
                    st.session_state.data["total_hours"] = total_hours
                    st.session_state.data["semester"] = semester

                    # 1단계: 필요성, 개요만 생성
                    basic_info = generate_content(1, st.session_state.data, vector_store)
                    if basic_info:
                        st.session_state.data.update(basic_info)  # {'necessity': "...", 'overview': "..."}
                        st.success("기본 정보 생성 완료.")
                        st.session_state.generated_step_1 = True
            else:
                st.error("모든 필수 항목을 입력해주세요.")

  
    if 'generated_step_1' in st.session_state:
        with st.form("edit_basic_info_form"):
            st.markdown("#### 생성된 내용 수정")
            necessity = st.text_area(
                "활동의 필요성",
                value=st.session_state.data.get('necessity', ''),
                height=150
            )
            overview = st.text_area(
                "활동 개요",
                value=st.session_state.data.get('overview', ''),
                height=150
            )

            submit_button_edit = st.form_submit_button("수정사항 저장 및 다음 단계로", use_container_width=True)

        if submit_button_edit:
            with st.spinner("수정사항 저장 중..."):
                st.session_state.data["necessity"] = necessity
                st.session_state.data["overview"] = overview
                del st.session_state.generated_step_1
                st.success("수정사항 저장 완료.")
                st.session_state.step = 2
                st.rerun()

    return False


def show_step_2_approval(vector_store):
    st.markdown("<div class='step-header'><h3>2단계: 자율시간 승인 신청서 다운로드</h3></div>", unsafe_allow_html=True)
    st.info("입력한 기본 정보를 바탕으로 승인 신청서 엑셀 파일을 생성합니다.")

    fields = ["학교급", "대상 학년", "총 차시", "운영 학기", "연계 교과", "활동명", "요구사항", "필요성", "개요"]

    selected_fields = st.multiselect(
        "다운로드할 항목 선택:",
        options=fields,
        default=fields,
        help="원하는 항목만 선택하여 파일에 포함할 수 있습니다."
    )
    if selected_fields:
        excel_data = create_approval_excel_document(selected_fields)
        st.download_button(
            "자율시간 승인 신청서 다운로드", excel_data,
            file_name=f"{st.session_state.data.get('activity_name', '자율시간승인신청서')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.warning("최소 하나의 항목을 선택해주세요.")

    if st.button("다음 단계로", use_container_width=True):
        st.session_state.step = 3
        st.rerun()


def create_approval_excel_document(selected_fields):
    output = BytesIO()
    all_fields = {
        "학교급": st.session_state.data.get('school_type', ''),
        "대상 학년": ', '.join(st.session_state.data.get('grades', [])),
        "총 차시": st.session_state.data.get('total_hours', ''),
        "운영 학기": ', '.join(st.session_state.data.get('semester', [])),
        "연계 교과": ', '.join(st.session_state.data.get('subjects', [])),
        "활동명": st.session_state.data.get('activity_name', ''),
        "요구사항": st.session_state.data.get('requirements', ''),
        "필요성": st.session_state.data.get('necessity', ''),
        "개요": st.session_state.data.get('overview', '')
    }
    selected_data = {k: v for k, v in all_fields.items() if k in selected_fields}
    df = pd.DataFrame({
        "항목": list(selected_data.keys()),
        "내용": list(selected_data.values())
    })
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="자율시간 승인 신청서")
        worksheet = writer.sheets["자율시간 승인 신청서"]
        worksheet.set_column("A:A", 20)
        worksheet.set_column("B:B", 50)
    return output.getvalue()


def show_step_3(vector_store):
    st.markdown("<div class='step-header'><h3>3단계: 내용체계</h3></div>", unsafe_allow_html=True)

    if 'generated_step_2' not in st.session_state:
        with st.form("generate_4sets"):
            st.info("영역명, 핵심 아이디어, 내용 요소를 **4세트** 생성합니다.")
            submit_btn = st.form_submit_button("4세트 생성 및 다음 단계로", use_container_width=True)
        if submit_btn:
            with st.spinner("생성 중..."):
                content = generate_content(3, st.session_state.data, vector_store)
                if isinstance(content, list) and len(content) == 4:
                    st.session_state.data["content_sets"] = content
                    st.success("4세트 내용체계 생성 완료.")
                else:
                    st.warning("4세트 형태가 아닌 응답이 왔습니다. 기본값 사용.")
                    st.session_state.data["content_sets"] = []
                st.session_state.generated_step_2 = True
    else:
        content_sets = st.session_state.data.get("content_sets", [])
        if not content_sets:
            content_sets = []

        with st.form("edit_4sets_form"):
            st.markdown("#### 생성된 4세트 내용체계 수정")
            new_sets = []
            tabs = st.tabs([f"내용체계 {i+1}" for i in range(4)])
            for i, tab in enumerate(tabs):
                with tab:
                    if i < len(content_sets):
                        cset = content_sets[i]
                    else:
                        cset = {
                            "domain": "",
                            "key_ideas": [],
                            "content_elements": {
                                "knowledge_and_understanding": [],
                                "process_and_skills": [],
                                "values_and_attitudes": []
                            }
                        }
                    domain_input = st.text_input("영역명", value=cset.get("domain",""), key=f"domain_{i}")
                    ki_list = cset.get("key_ideas", [])
                    ki_text = "\n".join(ki_list) if ki_list else ""
                    ki_input = st.text_area("핵심 아이디어", value=ki_text, height=80, key=f"ki_{i}")

                    ce = cset.get("content_elements", {})
                    kua = ce.get("knowledge_and_understanding", [])
                    pns = ce.get("process_and_skills", [])
                    vat = ce.get("values_and_attitudes", [])

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("##### 지식·이해")
                        kua_text = "\n".join(kua) if kua else ""
                        kua_input = st.text_area("knowledge_and_understanding", value=kua_text, height=120, key=f"kua_{i}")
                    with col2:
                        st.markdown("##### 과정·기능")
                        pns_text = "\n".join(pns) if pns else ""
                        pns_input = st.text_area("process_and_skills", value=pns_text, height=120, key=f"pns_{i}")
                    with col3:
                        st.markdown("##### 가치·태도")
                        vat_text = "\n".join(vat) if vat else ""
                        vat_input = st.text_area("values_and_attitudes", value=vat_text, height=120, key=f"vat_{i}")

                    new_sets.append({
                        "domain": domain_input,
                        "key_ideas": [line.strip() for line in ki_input.split("\n") if line.strip()],
                        "content_elements": {
                            "knowledge_and_understanding": [line.strip() for line in kua_input.split("\n") if line.strip()],
                            "process_and_skills": [line.strip() for line in pns_input.split("\n") if line.strip()],
                            "values_and_attitudes": [line.strip() for line in vat_input.split("\n") if line.strip()]
                        }
                    })

            submit_edit = st.form_submit_button("4세트 저장 및 다음 단계로", use_container_width=True)

        if submit_edit:
            with st.spinner("저장 중..."):
                st.session_state.data["content_sets"] = new_sets
                combined_key_ideas = []
                for cset in new_sets:
                    combined_key_ideas.extend(cset.get("key_ideas", []))
                st.session_state.data["key_ideas"] = combined_key_ideas

                if new_sets:
                    st.session_state.data["domain"] = new_sets[0]["domain"]
                    st.session_state.data["content_elements"] = new_sets[0]["content_elements"]
                else:
                    st.session_state.data["domain"] = ""
                    st.session_state.data["content_elements"] = {}

                del st.session_state.generated_step_2
                st.success("4세트 내용 저장 완료. 모든 핵심아이디어를 합쳐서 반영했습니다.")
                st.session_state.step = 4
                st.rerun()
    return False


def make_code_prefix(grades, subjects, activity_name):
    """
    학년/교과/활동명을 바탕으로 성취기준 코드의 접두사(prefix)를 생성.
    예) 학년 "3학년", 과목 "과학", 활동명 "인공지능놀이터" -> "3과인공"
    """
    grade_part = ""
    if grades:
        grade_part = grades[0].replace("학년", "").replace("학년군","").strip()
    subject_part = ""
    if subjects:
        s = subjects[0]
        subject_part = s[0]
    act_part = ""
    if activity_name:
        act_part = activity_name[:2]
    code_prefix = f"{grade_part}{subject_part}{act_part}"
    return code_prefix


def show_step_4(vector_store):
    st.markdown("<div class='step-header'><h3>4단계: 성취기준 설정</h3></div>", unsafe_allow_html=True)
    code_prefix = make_code_prefix(
        st.session_state.data.get('grades', []),
        st.session_state.data.get('subjects', []),
        st.session_state.data.get('activity_name', '')
    )
    content_sets = st.session_state.data.get("content_sets", [])
    num_sets = len(content_sets)

    if 'generated_step_3' not in st.session_state:
        with st.form("standards_form"):
            st.info(f"내용체계 세트가 {num_sets}개 생성되었습니다. 따라서 성취기준도 {num_sets}개를 생성합니다.")
            submit_button = st.form_submit_button("생성 및 다음 단계로", use_container_width=True)
        if submit_button:
            with st.spinner("생성 중..."):
                standards = generate_content(4, st.session_state.data, vector_store)
                if isinstance(standards, list) and len(standards) == num_sets:
                    st.session_state.data['standards'] = standards
                    st.success(f"성취기준 {num_sets}개 생성 완료.")
                    st.session_state.generated_step_3 = True
                else:
                    st.warning(f"{num_sets}개 성취기준이 아니라 기본값 사용")
                    st.session_state.data['standards'] = []
                    st.session_state.generated_step_3 = True
    else:
        with st.form("edit_standards_form"):
            st.markdown("#### 생성된 성취기준 수정")
            edited_standards = []
            for i, standard in enumerate(st.session_state.data.get('standards', [])):
                st.markdown(f"##### 성취기준 {i+1}")
                code = st.text_input("성취기준 코드", value=standard['code'], key=f"std_code_{i}")
                description = st.text_area("성취기준 설명", value=standard['description'],
                                           key=f"std_desc_{i}", height=100)
                st.markdown("##### 수준별 성취기준 (상, 중, 하)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    a_desc = st.text_area("상(A) 수준",
                                          value=next((l['description'] for l in standard['levels'] if l['level'] == 'A'), ''),
                                          key=f"std_{i}_level_A", height=100)
                with col2:
                    b_desc = st.text_area("중(B) 수준",
                                          value=next((l['description'] for l in standard['levels'] if l['level'] == 'B'), ''),
                                          key=f"std_{i}_level_B", height=100)
                with col3:
                    c_desc = st.text_area("하(C) 수준",
                                          value=next((l['description'] for l in standard['levels'] if l['level'] == 'C'), ''),
                                          key=f"std_{i}_level_C", height=100)
                edited_standards.append({
                    "code": code,
                    "description": description,
                    "levels": [
                        {"level": "A", "description": a_desc},
                        {"level": "B", "description": b_desc},
                        {"level": "C", "description": c_desc}
                    ]
                })
                st.markdown("---")
            submit_button_edit = st.form_submit_button("수정사항 저장 및 다음 단계로", use_container_width=True)
        if submit_button_edit:
            with st.spinner("저장 중..."):
                st.session_state.data['standards'] = edited_standards
                del st.session_state.generated_step_3
                st.success("성취기준 저장 완료.")
                st.session_state.step = 5
                st.rerun()
    return False


def show_step_5(vector_store):
    """
    5단계: 교수학습 및 평가
    - generate_content(5)를 통해 'criteria_high','criteria_mid','criteria_low'로 받음
    - 수정 폼에서 '코드, 성취기준, 평가요소, 수업평가방법'을 한 행(row)에 배치
      그리고 '평가기준(상, 중, 하)'은 각각 별도 text_area로 세로(행)로 배치.
    """
    st.markdown("<div class='step-header'><h3>5단계: 교수학습 및 평가</h3></div>", unsafe_allow_html=True)

    
    if 'generated_step_4' not in st.session_state:
        with st.form("teaching_assessment_form"):
            st.info("교수학습방법 및 평가계획을 자동으로 생성합니다.")
            submit_button = st.form_submit_button("생성 및 다음 단계로", use_container_width=True)
        if submit_button:
            with st.spinner("생성 중..."):
                result = generate_content(5, st.session_state.data, vector_store)
                if result:
                    st.session_state.data["teaching_methods_text"] = result.get("teaching_methods_text", "")
                    st.session_state.data["assessment_plan"] = result.get("assessment_plan", [])
                    st.success("교수학습 및 평가 생성 완료.")
                else:
                    st.warning("교수학습 및 평가 생성 실패. 기본값 사용.")
                    st.session_state.data["teaching_methods_text"] = ""
                    st.session_state.data["assessment_plan"] = []
                st.session_state.generated_step_4 = True

    else:
        
        with st.form("edit_teaching_assessment_form"):
            st.markdown("#### 교수학습방법 ")
            teaching_methods_text = st.text_area(
                "교수학습방법",
                value=st.session_state.data.get("teaching_methods_text", ""),
                height=120,
                help="줄바꿈으로 여러 방법을 구분"
            )

            st.markdown("""
            ---
            #### 평가계획
            - 코드, 성취기준, 평가요소, 수업평가방법을 한 행에 배치  
            - 평가기준(상·중·하)는 세로(행)로 각각 배치  
            """)

            old_plan = st.session_state.data.get("assessment_plan", [])
            new_plan = []

            for i, ap in enumerate(old_plan):
                code = ap.get("code", "")
                desc = ap.get("description", "")
                elem = ap.get("element", "")
                meth = ap.get("method", "")
                crit_high = ap.get("criteria_high", "")
                crit_mid = ap.get("criteria_mid", "")
                crit_low = ap.get("criteria_low", "")

                st.markdown(f"##### 평가항목 {i+1}")

                # ▶ 첫 번째 행: (코드 + 성취기준) / 평가요소 / 수업평가방법
                row1_col1, row1_col2, row1_col3 = st.columns([2, 2, 2])
                with row1_col1:
                    # '코드'와 '성취기준'은 수정 없이 표시만
                    st.markdown(f"**코드**: `{code}`")
                    st.markdown(f"**성취기준**: {desc}")

                with row1_col2:
                    new_elem = st.text_area("평가요소", value=elem, key=f"elem_{code}", height=80)
                with row1_col3:
                    new_meth = st.text_area("수업평가방법", value=meth, key=f"meth_{code}", height=80)

                # ▶ 두 번째 행: 평가기준(상, 중, 하)를 각각 세로(행)로 배치
                st.markdown("**평가기준(상·중·하)**")
                crit_high_new = st.text_area(
                    "상(A) 수준 기준",
                    value=crit_high,
                    key=f"critH_{code}",
                    height=80
                )
                crit_mid_new = st.text_area(
                    "중(B) 수준 기준",
                    value=crit_mid,
                    key=f"critM_{code}",
                    height=80
                )
                crit_low_new = st.text_area(
                    "하(C) 수준 기준",
                    value=crit_low,
                    key=f"critL_{code}",
                    height=80
                )

                # 수정한 내용을 new_plan에 반영
                new_plan.append({
                    "code": code,
                    "description": desc,
                    "element": new_elem,
                    "method": new_meth,
                    "criteria_high": crit_high_new,
                    "criteria_mid": crit_mid_new,
                    "criteria_low": crit_low_new
                })

                st.markdown("---")

            submit_button_edit = st.form_submit_button("수정사항 저장 및 다음 단계로", use_container_width=True)

        if submit_button_edit:
            with st.spinner("수정사항 저장 중..."):
                st.session_state.data["teaching_methods_text"] = teaching_methods_text
                st.session_state.data["assessment_plan"] = new_plan
                del st.session_state.generated_step_4
                st.success("교수학습 및 평가 수정 완료.")
                st.session_state.step = 6
                st.rerun()

    return False



def generate_lesson_plans_all_at_once(total_hours, data, vector_store=None):
    
    all_lesson_plans = []
    progress_bar = st.progress(0)

    necessity = data.get('necessity', '')
    overview = data.get('overview', '')
    domain = data.get('domain', '')
    key_ideas = data.get('key_ideas', [])
    content_elements = data.get('content_elements', {})
    standards = data.get('standards', [])
    teaching_methods = data.get('teaching_methods', [])
    assessment_plan = data.get('assessment_plan', [])
    progress_bar = st.progress(0)
    progress_text = st.empty()

    
    doc_context = ""
    if vector_store:
        retriever = vector_store.as_retriever()
        query = "지도계획"  # 원하는 검색어
        relevant_docs = retriever.get_relevant_documents(query)
        doc_context = "\n\n".join(doc.page_content for doc in relevant_docs)
    
        chunk_prompt = f"""
아래 정보를 참고하여 **1차시부터 {total_hours}차시까지** 한 번에 모두 연결된 지도계획을 JSON으로 작성해주세요.
[검색된 문서에서 가져온 맥락]
        {doc_context}
[이전 단계 결과]
대상 학년 {', '.join(data.get('grades', []))}에 맞는 수준으로 작성해야 한다.
- 영역명: {domain}
- 핵심 아이디어: {key_ideas}
- 내용체계: {content_elements}
- 성취기준: {standards}
- 교수학습 방법: {teaching_methods}
- 평가계획: {assessment_plan}
- 활동명: {data.get('activity_name')}
- 요구사항: {data.get('requirements')}
각 차시는 다음 사항을 고려하여 작성:
1. 대상 학년: {', '.join(data.get('grades', []))}에 알맞은 수업계획 자성하기기
2. 명확한 학습주제 재미있고 문학적 표현으로 학습주제 설정
3. 구체적이고 학생활동 중심으로 진술하세요. ~~하기 형식으로 해주세요.
4. 실제 수업에 필요한 교수학습자료 명시
5. 이전 차시와의 연계성 고려
6. 문서에서 가져온 결과를 그대로 사용하지 않고 서술어 위주의 표현만 참고하여 맥락에 맞게 사용하기 
7. 아래 예시를 참고하여 작성해주세요.
8. 초등학교 3학년 4학년 수준에 맞는 내용으로 작성하여 주세요.
(예시)
학습주제: 질문에도 양심이 있다.
학습내용: 질문을 할 때 지켜야 할 약속 만들기
         수업 중 질문, 일상 속 질문 속에서 갖추어야 할 예절 알기

“추가 문장 없이 JSON만 보내라”
다음 JSON 형식으로 작성:
{{
  "lesson_plans": [
    {{
      "lesson_number": "차시번호",
      "topic": "학습주제",
      "content": "학습내용",
      "materials": "교수학습자료"
    }}
  ]
}}
"""
        messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=chunk_prompt)
    ]
    try:
        chat = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model="gpt-4o",  
            temperature=0.5,
            max_tokens=3000  # 충분히 큰 값
        )
        response = chat(messages)
        raw_text = response.content.strip().replace('```json','').replace('```','').strip()
        parsed = json.loads(raw_text)
        lesson_plans = parsed.get("lesson_plans", [])
        return lesson_plans
    except json.JSONDecodeError as e:
        st.error(f"JSON 파싱 오류: {e}")
        return []
    except Exception as e:
        st.error(f"전체 차시 계획 생성 중 오류: {e}")
        return []


def show_step_6(vector_store):
    total_hours = st.session_state.data.get('total_hours', 30)
    st.markdown(f"<div class='step-header'><h3>6단계: 차시별 지도계획 (총 {total_hours}차시)</h3></div>", unsafe_allow_html=True)

    if 'generated_step_5' not in st.session_state:
        with st.form("lesson_plans_form"):
            st.info(f"총 {total_hours}차시를 한 번에 생성합니다.")
            sb = st.form_submit_button("전체 차시 생성", use_container_width=True)
        if sb:
            with st.spinner("생성 중..."):
                lesson_plans = generate_lesson_plans_all_at_once(total_hours, st.session_state.data, vector_store)
                if lesson_plans:
                    st.session_state.data["lesson_plans"] = lesson_plans
                    st.success(f"{total_hours}차시 계획 생성 완료.")
                    st.session_state.generated_step_5 = True
    else:
        with st.form("edit_lesson_plans_form"):
            st.markdown("#### 생성된 차시별 계획 수정")
            lesson_plans = st.session_state.data.get('lesson_plans', [])
            edited_plans = []
            total_tabs = (total_hours + 9) // 10
            tabs = st.tabs([f"{i*10+1}~{min((i+1)*10, total_hours)}차시" for i in range(total_tabs)])
            for tab_idx, tab in enumerate(tabs):
                with tab:
                    start_idx = tab_idx * 10
                    end_idx = min(start_idx + 10, total_hours)
                    for i in range(start_idx, end_idx):
                        st.markdown(f"##### {i+1}차시")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            topic = st.text_input("학습주제", value=lesson_plans[i].get('topic', ''),
                                                  key=f"topic_{i}")
                            materials = st.text_input("교수학습자료", value=lesson_plans[i].get('materials', ''),
                                                      key=f"materials_{i}")
                        with col2:
                            content = st.text_area("학습내용", value=lesson_plans[i].get('content', ''),
                                                   key=f"content_{i}", height=100)
                        edited_plans.append({
                            "lesson_number": f"{i+1}",
                            "topic": topic,
                            "content": content,
                            "materials": materials
                        })
                        st.markdown("---")
            submit_button_edit = st.form_submit_button("수정사항 저장 및 다음 단계로", use_container_width=True)
        if submit_button_edit:
            with st.spinner("저장 중..."):
                st.session_state.data['lesson_plans'] = edited_plans
                del st.session_state.generated_step_5
                st.success("차시별 계획 수정 완료.")
                st.session_state.step = 7
                st.rerun()
    return False


def show_final_review(vector_store):
    st.title("최종 계획서 검토")
    try:
        data = st.session_state.data
        tabs = st.tabs(["기본정보", "내용체계", "성취기준", "교수학습 및 평가", "차시별계획"])

        # 1) 기본정보
        with tabs[0]:
            st.markdown("### 기본 정보")
            basic_info = {
                "학교급": data.get('school_type', ''),
                "대상 학년": ', '.join(data.get('grades', [])),
                "총 차시": f"{data.get('total_hours','')}차시",
                "주당 차시": f"{data.get('weekly_hours','')}차시",
                "운영 학기": ', '.join(data.get('semester', [])),
                "연계 교과": ', '.join(data.get('subjects', [])),
                "활동명": data.get('activity_name',''),
                "요구사항": data.get('requirements',''),
                "필요성": data.get('necessity',''),
                "개요": data.get('overview','')
            }
            for k,v in basic_info.items():
                st.markdown(f"**{k}**: {v}")

            # 원하는 단계로 돌아갈 수 있도록 버튼
            st.button("기본정보 수정하기", key="edit_basic_info",
                      on_click=lambda: set_step(1),
                      use_container_width=True)

        # 2) 내용체계 (4세트 모두 표시)
        with tabs[1]:
            st.markdown("### 내용체계 (4세트)")

            content_sets = data.get("content_sets", [])
            if not content_sets:
                st.warning("현재 저장된 내용체계가 없습니다.")
            else:
                for i, cset in enumerate(content_sets, start=1):
                    st.markdown(f"#### ▶ 내용체계 세트 {i}")
                    domain = cset.get("domain", "")
                    key_ideas = cset.get("key_ideas", [])
                    content_elements = cset.get("content_elements", {})

                    st.write(f"**영역명**: {domain}")
                    st.write("**핵심 아이디어**:")
                    if key_ideas:
                        for idea in key_ideas:
                            st.write(f"- {idea}")
                    else:
                        st.write("- (없음)")

                    st.write("**내용 요소**:")
                    kua = content_elements.get("knowledge_and_understanding", [])
                    pns = content_elements.get("process_and_skills", [])
                    vat = content_elements.get("values_and_attitudes", [])

                    st.markdown("- 지식·이해")
                    if kua:
                        for item in kua:
                            st.write(f"  - {item}")
                    else:
                        st.write("  - (없음)")

                    st.markdown("- 과정·기능")
                    if pns:
                        for item in pns:
                            st.write(f"  - {item}")
                    else:
                        st.write("  - (없음)")

                    st.markdown("- 가치·태도")
                    if vat:
                        for item in vat:
                            st.write(f"  - {item}")
                    else:
                        st.write("  - (없음)")

                    st.divider()

            # 내용체계 수정 단계(3단계)
            st.button("내용체계 수정하기",
                      key="edit_content_sets",
                      on_click=lambda: set_step(3),
                      use_container_width=True)

        # 3) 성취기준
        with tabs[2]:
            st.markdown("### 성취기준")
            for std in data.get("standards", []):
                st.markdown(f"**{std['code']}**: {std['description']}")
                st.markdown("##### 수준별 성취기준")
                for lv in std['levels']:
                    label_map = {"A":"상", "B":"중", "C":"하"}
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
            for ap in data.get("assessment_plan", []):
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

    
        with tabs[4]:
            st.markdown("### 차시별 계획")
            lesson_plans_df = pd.DataFrame(data.get('lesson_plans', []))
            if not lesson_plans_df.empty:
                st.dataframe(
                    lesson_plans_df,
                    column_config={
                        "lesson_number": "차시",
                        "topic": "학습주제",
                        "content": "학습내용",
                        "materials": "교수학습자료"
                    },
                    hide_index=True,
                    height=400
                )
            else:
                st.warning("차시별 계획이 없습니다.")

            st.button("차시별 계획 수정하기",
                      key="edit_lesson_plans",
                      on_click=lambda: set_step(6),
                      use_container_width=True)

        # 화면 아래에 다운로드 및 처음으로 돌아가기 버튼
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("모든 단계 수정하기", use_container_width=True):
                st.session_state.step = 1
                st.rerun()

        with col2:
            st.markdown("#### 원하는 항목만 선택하여 Excel 다운로드")
            available_sheets = ["기본정보", "내용체계", "성취기준", "교수학습 및 평가", "차시별계획"]
            selected_sheets = st.multiselect(
                "다운로드할 항목",
                options=available_sheets,
                default=available_sheets
            )
            if selected_sheets:
                excel_data = create_excel_document(selected_sheets)
                st.download_button(
                    "📥 Excel 다운로드",
                    excel_data,
                    file_name=f"{data.get('activity_name','학교자율시간계획서')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                st.warning("최소 한 개 이상의 항목을 선택해주세요.")

        with col3:
            if st.button("새로 만들기", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    except Exception as e:
        st.error(f"최종 검토 처리 중 오류: {str(e)}")


def create_excel_document(selected_sheets):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#E2E8F0',
            'border': 1,
            'text_wrap': True,
            'align': 'center',
            'valign': 'vcenter'
        })
        content_format = workbook.add_format({
            'text_wrap': True,
            'valign': 'top',
            'border': 1
        })
        data = st.session_state.data

        if "기본정보" in selected_sheets:
            basic_info = pd.DataFrame([{
                '학교급': data.get('school_type', ''),
                '대상학년': ', '.join(data.get('grades', [])),
                '총차시': data.get('total_hours', ''),
                '주당차시': data.get('weekly_hours', ''),
                '운영 학기': ', '.join(data.get('semester', [])),
                '연계 교과': ', '.join(data.get('subjects', [])),
                '활동명': data.get('activity_name', ''),
                '요구사항': data.get('requirements', ''),
                '필요성': data.get('necessity', ''),
                '개요': data.get('overview', '')
            }])
            basic_info.T.to_excel(writer, sheet_name='기본정보', header=['내용'])
            worksheet = writer.sheets['기본정보']
            for idx, col in enumerate(basic_info.T.index, 1):
                worksheet.set_column(idx, idx, 30, content_format)

        if "내용체계" in selected_sheets:
            content_sets = data.get("content_sets", [])
            if not content_sets:
                df_empty = pd.DataFrame([{"구분": "내용체계 없음", "내용": ""}])
                df_empty.to_excel(writer, sheet_name='내용체계', index=False)
                worksheet = writer.sheets['내용체계']
                worksheet.set_column('A:A', 20, content_format)
                worksheet.set_column('B:B', 80, content_format)
            else:
                rows = []
                for idx, cset in enumerate(content_sets, start=1):
                    domain = cset.get("domain", "")
                    key_ideas = cset.get("key_ideas", [])
                    ce = cset.get("content_elements", {})

                    rows.append({
                        "구분": f"영역명 (세트{idx})",
                        "내용": domain
                    })

                    for idea in key_ideas:
                        rows.append({
                            "구분": f"핵심 아이디어 (세트{idx})",
                            "내용": idea
                        })

                    for item in ce.get("knowledge_and_understanding", []):
                        rows.append({
                            "구분": f"지식·이해 (세트{idx})",
                            "내용": item
                        })

                    for item in ce.get("process_and_skills", []):
                        rows.append({
                            "구분": f"과정·기능 (세트{idx})",
                            "내용": item
                        })

                    for item in ce.get("values_and_attitudes", []):
                        rows.append({
                            "구분": f"가치·태도 (세트{idx})",
                            "내용": item
                        })

                df_goals = pd.DataFrame(rows)
                df_goals.to_excel(writer, sheet_name='내용체계', index=False)
                worksheet = writer.sheets['내용체계']
                worksheet.set_column('A:A', 25, content_format)
                worksheet.set_column('B:B', 80, content_format)

        if "성취기준" in selected_sheets:
            standards_data = []
            for std in data.get('standards', []):
                for level in std['levels']:
                    label_map = {"A": "상", "B": "중", "C": "하"}
                    label = label_map.get(level['level'], level['level'])
                    standards_data.append({
                        '성취기준코드': std['code'],
                        '성취기준설명': std['description'],
                        '수준': label,
                        '수준별설명': level['description']
                    })
            df_std = pd.DataFrame(standards_data)
            df_std.to_excel(writer, sheet_name='성취기준', index=False)
            worksheet = writer.sheets['성취기준']
            worksheet.set_column('A:A', 15, content_format)
            worksheet.set_column('B:B', 50, content_format)
            worksheet.set_column('C:C', 10, content_format)
            worksheet.set_column('D:D', 60, content_format)

        if "교수학습 및 평가" in selected_sheets:
            sheet_rows = []
            methods_text = data.get("teaching_methods_text", "").strip()
            if methods_text:
                lines = methods_text.split('\n')
                for line in lines:
                    if line.strip():
                        sheet_rows.append({
                            "유형": "교수학습방법",
                            "코드": "",
                            "성취기준": "",
                            "평가요소": "",
                            "수업평가방법": line.strip(),
                            "상기준": "",
                            "중기준": "",
                            "하기준": ""
                        })

            # assessment_plan: code, description, element, method, criteria_high, criteria_mid, criteria_low
            for ap in data.get('assessment_plan', []):
                sheet_rows.append({
                    "유형": "평가계획",
                    "코드": ap.get("code",""),
                    "성취기준": ap.get("description",""),
                    "평가요소": ap.get("element",""),
                    "수업평가방법": ap.get("method",""),
                    "상기준": ap.get("criteria_high",""),
                    "중기준": ap.get("criteria_mid",""),
                    "하기준": ap.get("criteria_low","")
                })

            df_methods = pd.DataFrame(sheet_rows)
            df_methods.to_excel(writer, sheet_name='교수학습및평가', index=False)
            worksheet = writer.sheets['교수학습및평가']
            worksheet.set_column('A:A', 14, content_format)
            worksheet.set_column('B:B', 14, content_format)
            worksheet.set_column('C:C', 30, content_format)
            worksheet.set_column('D:D', 30, content_format)
            worksheet.set_column('E:E', 30, content_format)
            worksheet.set_column('F:F', 30, content_format)
            worksheet.set_column('G:G', 30, content_format)
            worksheet.set_column('H:H', 30, content_format)

        if "차시별계획" in selected_sheets:
            df_lessons = pd.DataFrame(data.get('lesson_plans', []))
            if not df_lessons.empty:
                df_lessons.columns = ['차시', '학습주제', '학습내용', '교수학습자료']
                df_lessons.to_excel(writer, sheet_name='차시별계획', index=False)
                worksheet = writer.sheets['차시별계획']
                worksheet.set_column('A:A', 10, content_format)
                worksheet.set_column('B:B', 30, content_format)
                worksheet.set_column('C:C', 80, content_format)
                worksheet.set_column('D:D', 50, content_format)

        for sheet in writer.sheets.values():
            sheet.set_default_row(30)
            sheet.set_row(0, 40)

    return output.getvalue()


def set_step(step_number):
    st.session_state.step = step_number


def show_chatbot(vector_store):
    st.sidebar.markdown("## 학교자율시간 교육과정 설계 챗봇")

    st.sidebar.markdown("**추천 질문:**")
    recommended_questions = [
        "초등학교 3학년 학교자율시간의 활동명 10가지만 제시하여 주세요.",
        "초등학교 6학년 세계요리탐험에 알맞은 수업지도 계획을 작성해주세요.",
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
            retriever = vector_store.as_retriever()
            results = retriever.get_relevant_documents(user_input)
            context = "\n\n".join([doc.page_content for doc in results])
            prompt = f"""당신은 귀여운 친구 캐릭터 두 명, '🐰 토끼'와 '🐻 곰돌이'입니다.
두 캐릭터는 협력하여 학교자율시간 관련 질문에 대해 번갈아 가며 귀엽고 친근한 말투로 답변합니다.
지침
- 문서에 제시된 개념/표현을 최대한 반영
- 문서와 모순되는 내용 쓰지 말기
- 문서에 없는 내용은 최소화
- 문서에서 적합한 표현이 있으면 그대로 활용
질문: {user_input}
관련 정보: {context}
답변:"""
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            chat = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                model="gpt-4o",
                temperature=0.7,
                max_tokens=2000
            )
            response = chat(messages)
            answer = response.content.strip()
            st.sidebar.markdown("**🤖 답변:**")
            sidebar_typewriter_effect("🤖 " + answer, delay=0.001)
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
        if 'data' not in st.session_state:
            st.session_state.data = {}
        if 'step' not in st.session_state:
            st.session_state.step = 1
        st.title("학교자율시간 올인원")

        vector_store = setup_vector_store()
        if not vector_store:
            st.error("문서 임베딩 실패. `documents/` 폴더를 확인해주세요.")
            return

        left_col = st.container()
        with left_col:
            show_progress()
            step_functions = {
                1: show_step_1,
                2: show_step_2_approval,
                3: show_step_3,
                4: show_step_4,
                5: show_step_5,
                6: show_step_6,
                7: show_final_review
            }
            current_step = st.session_state.step
            step_function = step_functions.get(current_step)
            if step_function:
                step_function(vector_store)
            else:
                st.error("잘못된 단계입니다.")

        # 사이드바 챗봇
        show_chatbot(vector_store)

    except Exception as e:
        st.error(f"애플리케이션 실행 중 오류: {e}")
        if st.button("처음부터 다시 시작"):
            st.session_state.clear()
            st.rerun()


if __name__ == "__main__":
    main()
st.markdown(
    """
    <div class="footer" style="text-align:center; margin-top:20px;">
        <!-- width를 원하는 픽셀(px) 혹은 퍼센트(%)로 조정 -->
        <img src="https://huggingface.co/spaces/powerwarez/gailabicon/resolve/main/gailab06.png"
             alt="icon"
             style="width:80px; height:auto;">
        <p>제작자: 제작: 경상북도교육청 인공지능연구소(GAI LAB) 교사 서혁수</p>
    </div>
    """,
    unsafe_allow_html=True
)

