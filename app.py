import os
import streamlit as st
import pandas as pd
from io import BytesIO
import json
import time

# --------------------------- LangChain ---------------------------
from langchain.prompts import ChatPromptTemplate
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.document_loaders import UnstructuredPDFLoader
# ---------------------------------------------------------------

# 0. OpenAI 클라이언트 초기화 & 시스템 프롬프트
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
if not OPENAI_API_KEY:
    st.error("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 확인하세요.")
    st.stop()

SYSTEM_PROMPT = """한국의 초등학교 2022 개정 교육과정 전문가입니다.
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

# --------------------------- 추가 기능 ---------------------------
def sidebar_typewriter_effect(text, delay=0.001):
    """사이드바에 한 글자씩 타이핑되듯 출력"""
    placeholder = st.sidebar.empty()
    output = ""
    for char in text:
        output += char
        placeholder.markdown(output)
        time.sleep(delay)
    return output
# --------------------------- 추가 기능 끝 ---------------------------

# 1. 페이지 기본 설정
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
        }
        .step-header {
            background-color: #f8fafc;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

# 2. 진행 상황 표시 (계획서 생성기 전용)
def show_progress():
    current_step = st.session_state.get('step', 1)
    steps = ["기본정보", "승인 신청서 다운로드", "내용체계", "성취기준", "교수학습 및 평가", "차시별계획", "최종 검토"]

    st.markdown("""
        <style>
        .step-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 1rem 0;
            flex-direction: row;
            width: 100%;
            padding: 20px;
        }
        .step-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            z-index: 2;
        }
        .step-circle {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-bottom: 8px;
        }
        .step-active {
            background-color: #3b82f6;
            color: white;
            box-shadow: 0 0 10px rgba(59,130,246,0.5);
            transform: scale(1.1);
            transition: all 0.3s ease;
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
        }
        .step-line {
            height: 4px;
            flex: 1;
            background-color: #e5e7eb;
            margin: 0 10px;
            position: relative;
            top: -25px;
            z-index: 1;
        }
        .step-line-completed {
            background-color: #10b981;
        }
        .step-line-active {
            background-color: #3b82f6;
        }
        .step-container-outer {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            padding: 10px 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    html = '<div class="step-container-outer"><div class="step-container">'
    for i, step in enumerate(steps, 1):
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
                <div class="step-label">{step}</div>
            </div>
        '''
        if i < len(steps):
            if i < current_step:
                line_style = "step-line-completed"
            elif i == current_step:
                line_style = "step-line-active"
            else:
                line_style = "step-line-pending"
            html += f'<div class="step-line {line_style}"></div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

# 3. 벡터 데이터베이스 설정 (문서 임베딩)
@st.cache_resource(show_spinner="벡터 스토어 로딩 중...")
def setup_vector_store():
    try:
        index_dir = "faiss_index"
        if os.path.exists(index_dir) and os.path.isdir(index_dir):
            st.success("기존 벡터 DB(FAISS 인덱스)를 로드합니다.")
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
                    # pdf인 경우 UnstructuredPDFLoader
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

# 4. 기본 콘텐츠 함수: 단계별 내용 생성
def generate_content(step, data, vector_store):
    try:
        # ---------------------- (1) 벡터 스토어 검색어(query) 수정 ----------------------
        # step 3 => "내용체계"
        # step 4 => "성취기준"
        # step 5 => "교수학습 및 평가"
        context = ""
        if step >= 3 and vector_store:
            retriever = vector_store.as_retriever()
            query_map = {
                3: "내용체계",
                4: "성취기준",
                5: "교수학습 및 평가"
            }
            query = query_map.get(step, "")
            if query:
                retrieved_docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # (2) 이전 단계 결과
        necessity = data.get('necessity', '')
        overview = data.get('overview', '')
        characteristics = data.get('characteristics', '')
        standards = data.get('standards', [])
        teaching_methods = data.get('teaching_methods', [])
        assessment_plan = data.get('assessment_plan', [])
        content_sets = data.get("content_sets", [])
        num_sets = len(content_sets)

        # (3) 단계별 프롬프트 - 키 3=내용체계, 4=성취기준, 5=교수학습 및 평가
        step_prompts = {
            1: f"""학교자율시간 활동의 기본 정보를 작성해주세요.

활동명: {data.get('activity_name')}
요구사항: {data.get('requirements')}
학교급: {data.get('school_type')}
대상 학년: {', '.join(data.get('grades', []))}
연계 교과: {', '.join(data.get('subjects', []))}
총 차시: {data.get('total_hours')}차시, 주당 {data.get('weekly_hours')}차시
운영 학기: {', '.join(data.get('semester', []))}

아래 예시와 같이, 주어진 **활동명에 종속**되어 결과물이 도출되도록 
'필요성', '개요', '성격'을 작성해 주세요.
지침
1. 필요성은 예시의 2~3배 분량으로 작성해주세요.
2. 개요는 ()로 성격, 목표, 주요 내용을 구분해주세요
[예시]
필요성:
 • 불확실한 미래사회를 살아갈 학생들에게 필수적 요소인 디지털 기기의 바른 이해와 사용법에 대한 학습이 필요
 • 디지털 기기 활용뿐 아니라 디지털 윤리에 관한 학습을 통해 디지털 리터러시와 책임감 있는 디지털 시민으로서의 역량 함양 필요

개요:
 <성격>
 • 디지털 기기 사용 경험을 바탕으로, 디지털 기술의 원리와 활용, 윤리적 문제점을 탐구하고, 안전하고 책임감 있는 디지털 시민으로 성장할 수 있도록 돕고,
   디지털 기술의 사회적 영향과 윤리적 책임을 고민하며 미래 사회를 준비하는 데 필요한 역량을 함양한다.
 <목표>
 • 디지털 기기의 작동 원리와 활용 방법을 이해한다.
 • 디지털 기기를 안전하고 책임감 있게 사용하는 방법을 익힌다.
 • 디지털 세상의 윤리적 문제에 대한 인식을 높이고 올바른 태도를 형성한다.
 <주요 내용>
 • 디지털 기기 작동 원리 및 간단한 프로그래밍
 • 디지털 기기를 활용한 다양한 창작 활동
 • 디지털 시민으로서 가져야 할 올바른 디지털 윤리

성격:
 • 위 개요의 <성격> 부분을 참고하여, 주어진 활동명에 맞는 활동 성격을 작성

다음 JSON 형식으로 작성:
{{
  "necessity": "작성된 필요성 내용",
  "overview": "작성된 개요 내용",
  "characteristics": "작성된 성격 내용"
}}
""",
            # 3단계: 내용체계
            3: f"""{context}
이전 단계 결과:
필요성: {necessity}
개요: {overview}
성격: {characteristics}
아래 예시를 참고하여, **'영역명(domain)', '핵심 아이디어(key_ideas)', '내용 요소(content_elements)'**를 JSON 구조로 작성해주세요. 
핵심아이디어는 IB교육에서 이야기 하는 빅아이디어와 같은 거야. 학생들이 도달 할 수 있는 일반화된 이론이야 예시처럼 문장으로 진술해주세요.
'content_elements'에는 **'knowledge_and_understanding'(지식·이해), 'process_and_skills'(과정·기능), 'values_and_attitudes'(가치·태도)**가 반드시 포함되어야 합니다.
'영역명(domain)', '핵심 아이디어(key_ideas)', '내용 요소(content_elements)'(지식·이해 / 과정·기능 / 가치·태도)
4개 세트를 생성...
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
이전 단계 결과(내용체계):
{content_sets}

총 {num_sets}개 내용체계 세트가 생성되었으므로, 성취기준도 {num_sets}개 생성.
(각 code, description, levels(A/B/C) )
지침
1. 성취기준코드는 입력된 대상 학년,연계 교과, 활동명(2글자 줄이기) 순이야 
(예시)4과텃밭-01 
3. 성취기준코드는 입력된 대상 학년, 연계 교과, 활동명과 일치하도록 구성해야 한다. 
4. 성취기준은 내용체계표와 내용이 비슷하고 문장의 형식은 아래와 같아
(예시)
[4사세계시민-01] 글을 읽고 지구촌의 여러 문제를 이해하고 생각한다.
[4사세계시민-02] 보편적인 핵심 가치를 생각하며 문제를 이해한다.
[4사세계시민-03] 지구촌의 여러 문제를 다양한 관점에서 사고한다.
[4사세계시민-04] 친구들과 상호작용하며 사회문제에 대한 나의 생각을 이야기한다.
[4사세계시민-05] 사회문제에 대한 자신과 타인의 관점을 파악하고 존중한다.
[4사세계시민-06] 타인과 소통하고 협력하며 세계시민의 자질을 기른다.
JSON 예시:
[
  {{
    "code": "기준코드",
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
            # 5단계: 교수학습 및 평가
             5: f"""
(5단계) 교수학습 및 평가
이전 단계(성취기준): {standards}
평가요소, 수업방법평가, 평가기준은 예시문을 참고해서 작성해주세요요
<예시>
평가요소
 • 주어진 이야기의 흐름에 맞게 이어질 내용을 자신이 선택한 표현 방법으로 친구들에게 발표하기
수업평가방법법
 • [창의성 계발 수업]
 • 주어진 이야기를 읽고 이어질 내용을 말, 글, 춤, 노래, 그림 등 나의 개성이 드러나는 방법으로 친구들 앞에서 발표함. [구술]
평가기준
  • 주어진 이야기를 자신의 개성에 맞는 다양한 표현 방법을 활용하여 친구들 앞에서 발표한다. 
"teaching_methods_text"교수학습도 예시문을 참고해서 작성하여 주세요
<예시>
• 인간 활동으로 발생한 환경 영향의 긍정적인 사례와 부정적인 사례를 균형적으로 탐구하여 인간과 환경에 대한 다양한 측면을 이해하도록 한다.
• 다양한 사례를 통하여 환경오염의 현상을 이해하도록 지도하고 지속가능한 발전으로 이어질 수 있도록 내면화에 노력한다. 
• 학교나 지역의 다양한 체험활동 장소와 주제에 따른 계절을 고려하여 학습계획을 세워 학습을 진행한다. 
• 탐구 및 활동 시에는 사전 준비와 안전교육 등을 통하여 탐구과정에서 발생할 수 있는 안전사고를 예방하도록 한다. 
"teaching_methods_text": 문자열
"assessment_plan": [
  {{
    "code": "성취기준코드(4단계)",
    "description": "성취기준문장(4단계)",
    "element": "평가요소",
    "method": "수업평가방법",
    "criteria": "평가기준"
  }},
  ...
]
"""
        }

        # 2단계/6단계/7단계는 LLM 사용 안 함 → 바로 {}
        if step in [2,6,7]:
            return {}

        prompt = step_prompts.get(step, "")
        if not prompt:
            return {}

        # LLM 호출
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt + "\n\n(위 형식으로 JSON만 반환)"),
        ]
        chat = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1800
        )
        response = chat(messages)
        raw_text = response.content.strip().replace('```json','').replace('```','').strip()

        # JSON 파싱
        try:
            parsed = json.loads(raw_text)
            if step == 5:
                # 검증
                if "teaching_methods_text" not in parsed or "assessment_plan" not in parsed:
                    raise ValueError("5단계: 'teaching_methods_text', 'assessment_plan' 모두 필요.")
                for ap in parsed["assessment_plan"]:
                    for field in ["code","description","element","method","criteria"]:
                        if field not in ap:
                            raise ValueError(f"assessment_plan 항목에 '{field}' 누락")
            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            st.warning(f"JSON 파싱 오류(단계 {step}): {e} => 기본값 사용")
            return {}

    except Exception as e:
        st.error(f"generate_content({step}) 중 오류: {e}")
        return {}

# 5. 단계별 UI 함수들
def show_step_1(vector_store):
    st.markdown("<div class='step-header'><h3>1단계: 기본 정보</h3></div>", unsafe_allow_html=True)
    if 'generated_step_1' not in st.session_state:
        with st.form("basic_info_form"):
            school_type = st.radio("학교급", ["초등학교", "중학교"], horizontal=True, key="school_type_radio")
            col1, col2 = st.columns(2)
            with col1:
                total_hours = st.number_input("총 차시", min_value=1, max_value=68,
                                               value=st.session_state.data.get('total_hours', 34),
                                               help="총 차시 입력 (최대 68차시)")
                weekly_hours = st.number_input("주당 차시", min_value=1, max_value=2,
                                                value=st.session_state.data.get('weekly_hours', 1),
                                                help="주당 차시 입력")
            with col2:
                semester = st.multiselect("운영 학기", ["1학기", "2학기"],
                                          default=st.session_state.data.get('semester', ["1학기"]))
            st.markdown("#### 학년 선택")
            if school_type == "초등학교":
                grades = st.multiselect("학년", ["3학년", "4학년", "5학년", "6학년"],
                                        default=st.session_state.data.get('grades', []))
                subjects = st.multiselect("교과", ["국어", "수학", "사회", "과학", "영어", "음악", "미술", "체육", "실과", "도덕"],
                                          default=st.session_state.data.get('subjects', []))
            else:
                grades = st.multiselect("학년", ["1학년", "2학년", "3학년"],
                                        default=st.session_state.data.get('grades', []))
                subjects = st.multiselect("교과", ["국어", "수학", "사회/역사", "과학/기술", "영어", "음악", "미술", "체육", "정보", "도덕"],
                                          default=st.session_state.data.get('subjects', []))
            col1, col2 = st.columns(2)
            with col1:
                activity_name = st.text_input("활동명",
                                              value=st.session_state.data.get('activity_name', ''),
                                              placeholder="예: 인공지능 놀이터")
            with col2:
                requirements = st.text_area("요구사항",
                                             value=st.session_state.data.get('requirements', ''),
                                             placeholder="예: 디지털 리터러시 강화 필요",
                                             height=100)
            submit_button = st.form_submit_button("정보 생성 및 다음 단계로", use_container_width=True)
        if submit_button:
            if activity_name and requirements and grades and subjects and semester:
                with st.spinner("정보 생성 중..."):
                    st.session_state.data.update({
                        'school_type': school_type,
                        'grades': grades,
                        'subjects': subjects,
                        'activity_name': activity_name,
                        'requirements': requirements,
                        'total_hours': total_hours,
                        'weekly_hours': weekly_hours,
                        'semester': semester
                    })
                    # step=1 => generate_content(1, ...)
                    basic_info = generate_content(1, st.session_state.data, vector_store)
                    if basic_info:
                        st.session_state.data.update(basic_info)
                        st.success("기본 정보 생성 완료.")
                        st.session_state.generated_step_1 = True
            else:
                st.error("모든 필수 항목을 입력해주세요.")
    if 'generated_step_1' in st.session_state:
        with st.form("edit_basic_info_form"):
            st.markdown("#### 생성된 내용 수정")
            necessity = st.text_area("활동의 필요성",
                                     value=st.session_state.data.get('necessity', ''),
                                     height=150,
                                     key="necessity_textarea")
            overview = st.text_area("활동 개요",
                                    value=st.session_state.data.get('overview', ''),
                                    height=150,
                                    key="overview_textarea")
            characteristics = st.text_area("활동의 성격",
                                           value=st.session_state.data.get('characteristics', ''),
                                           height=150,
                                           key="characteristics_textarea")
            submit_button_edit = st.form_submit_button("수정사항 저장 및 다음 단계로", use_container_width=True)
        if submit_button_edit:
            with st.spinner("수정사항 저장 중..."):
                st.session_state.data.update({
                    'necessity': necessity,
                    'overview': overview,
                    'characteristics': characteristics
                })
                del st.session_state.generated_step_1
                st.success("수정사항 저장 완료.")
                st.session_state.step = 2
                st.rerun()
    return False

# 2단계: 자율시간 승인 신청서 다운로드 (LLM 사용 안 함)
def show_step_2_approval(vector_store):
    st.markdown("<div class='step-header'><h3>2단계: 자율시간 승인 신청서 다운로드</h3></div>", unsafe_allow_html=True)
    st.info("입력한 기본 정보를 바탕으로 승인 신청서 엑셀 파일을 생성합니다.")
    fields = ["학교급", "대상 학년", "총 차시", "주당 차시", "운영 학기", "연계 교과", "활동명", "요구사항", "필요성", "개요", "성격"]
    selected_fields = st.multiselect("다운로드할 항목 선택:", options=fields, default=fields,
                                     help="원하는 항목만 선택하여 파일에 포함할 수 있습니다.")
    if selected_fields:
        excel_data = create_approval_excel_document(selected_fields)
        st.download_button("자율시간 승인 신청서 다운로드", excel_data,
                           file_name=f"{st.session_state.data.get('activity_name', '자율시간승인신청서')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
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
        "주당 차시": st.session_state.data.get('weekly_hours', ''),
        "운영 학기": ', '.join(st.session_state.data.get('semester', [])),
        "연계 교과": ', '.join(st.session_state.data.get('subjects', [])),
        "활동명": st.session_state.data.get('activity_name', ''),
        "요구사항": st.session_state.data.get('requirements', ''),
        "필요성": st.session_state.data.get('necessity', ''),
        "개요": st.session_state.data.get('overview', ''),
        "성격": st.session_state.data.get('characteristics', '')
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

# 3단계: 영역/핵심아이디어/내용요소 입력 및 생성 => generate_content(3, ...)
def show_step_3(vector_store):
    st.markdown("<div class='step-header'><h3>3단계: 내용체계</h3></div>", unsafe_allow_html=True)

    if 'generated_step_2' not in st.session_state:
        # 4세트 생성 폼
        with st.form("generate_4sets"):
            st.info("영역명, 핵심 아이디어, 내용 요소를 **4세트** 생성합니다.")
            submit_btn = st.form_submit_button("4세트 생성 및 다음 단계로", use_container_width=True)
        if submit_btn:
            with st.spinner("생성 중..."):
                # 변경: generate_content(2, ...) -> generate_content(3, ...)
                content = generate_content(3, st.session_state.data, vector_store)
                if isinstance(content, list) and len(content) == 4:
                    st.session_state.data["content_sets"] = content
                    st.success("4세트 내용체계 생성 완료.")
                else:
                    st.warning("4세트 형태가 아닌 응답이 왔습니다. 기본값 사용.")
                    st.session_state.data["content_sets"] = []
                st.session_state.generated_step_2 = True
    else:
        # 생성된 content_sets 편집
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
                # 4세트 각각의 key_ideas를 합쳐서 st.session_state.data["key_ideas"] 에 저장
                combined_key_ideas = []
                for cset in new_sets:
                    combined_key_ideas.extend(cset.get("key_ideas", []))

                st.session_state.data["key_ideas"] = combined_key_ideas

                # 첫 번째 세트의 domain, content_elements 만 대표로 사용
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

# 4단계: 성취기준 설정 => generate_content(4, ...)
def show_step_4(vector_store):
    st.markdown("<div class='step-header'><h3>4단계: 성취기준 설정</h3></div>", unsafe_allow_html=True)

    content_sets = st.session_state.data.get("content_sets", [])
    num_sets = len(content_sets)

    if 'generated_step_3' not in st.session_state:
        with st.form("standards_form"):
            st.info(f"내용체계 세트가 {num_sets}개 생성되었습니다. 따라서 성취기준도 {num_sets}개를 생성합니다.")
            submit_button = st.form_submit_button("생성 및 다음 단계로", use_container_width=True)
        if submit_button:
            with st.spinner("생성 중..."):
                # 변경: generate_content(3, ...) -> generate_content(4, ...)
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

# 5단계: 교수학습 및 평가 => generate_content(5, ...)
def show_step_5(vector_store):
    """
    5단계: 교수학습 및 평가
      - teaching_methods_text: 문자열(줄바꿈 구분)
      - assessment_plan: list of { code, description, element, method, criteria }
        code: 4단계 성취기준코드 (read-only)
        description: 4단계 성취기준문장 (read-only)
        element/method/criteria: 수정 가능
    """
    st.markdown("<div class='step-header'><h3>5단계: 교수학습 및 평가</h3></div>", unsafe_allow_html=True)

    if 'generated_step_4' not in st.session_state:
        # 아직 생성 안됨 => LLM 호출로 생성
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
                    st.warning("교수학습 및 평가 생성 실패. 기본값 사용")
                    st.session_state.data["teaching_methods_text"] = ""
                    st.session_state.data["assessment_plan"] = []
                st.session_state.generated_step_4 = True
    else:
        # 이미 생성됨 => 수정 폼
        with st.form("edit_teaching_assessment_form"):
            st.markdown("#### 교수학습방법 (여러 개를 줄바꿈으로 입력)")
            teaching_methods_text = st.text_area(
                "교수학습방법",
                value=st.session_state.data.get("teaching_methods_text",""),
                height=120,
                help="줄바꿈으로 여러 방법을 구분"
            )

            st.markdown("#### 평가계획: (성취기준코드,성취기준문장) + 평가요소,평가방법,평가기준")
            old_plan = st.session_state.data.get("assessment_plan", [])
            new_plan = []
            for i, ap in enumerate(old_plan):
                code = ap.get("code","")
                desc = ap.get("description","")
                elem = ap.get("element","")
                meth = ap.get("method","")
                crit = ap.get("criteria","")

                col1, col2, col3, col4, col5 = st.columns([1.5, 2, 2, 2, 2])
                with col1:
                    st.markdown(f"**코드**: {code}")
                with col2:
                    st.markdown(f"**문장**: {desc}")
                with col3:
                    new_elem = st.text_area("평가요소", value=elem, key=f"elem_{code}", height=100)
                with col4:
                    new_meth = st.text_area("평가방법", value=meth, key=f"meth_{code}", height=100)
                with col5:
                    new_crit = st.text_area("평가기준", value=crit, key=f"crit_{code}", height=100)

                new_plan.append({
                    "code": code,
                    "description": desc,
                    "element": new_elem,
                    "method": new_meth,
                    "criteria": new_crit
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

# 6단계: 차시별 지도계획 생성 (generate_lesson_plans_in_chunks)
def generate_lesson_plans_in_chunks(total_hours, data, chunk_size=10, vector_store=None):
    all_lesson_plans = []
    progress_bar = st.progress(0)

    # 이전 단계 데이터
    necessity = data.get('necessity', '')
    overview = data.get('overview', '')
    characteristics = data.get('characteristics', '')
    domain = data.get('domain', '')
    key_ideas = data.get('key_ideas', [])
    content_elements = data.get('content_elements', {})
    standards = data.get('standards', [])
    teaching_methods = data.get('teaching_methods', [])
    assessment_plan = data.get('assessment_plan', [])

    for start in range(0, total_hours, chunk_size):
        end = min(start + chunk_size, total_hours)
        progress_bar.progress(int((start / total_hours) * 100))
        st.write(f"{start+1}~{end}차시 계획 생성 중...")

        chunk_prompt = f"""
다음 정보를 바탕으로 {start+1}차시부터 {end}차시까지의 지도계획을 JSON으로 작성해주세요.

[이전 단계 결과]
- 필요성: {necessity}
- 개요: {overview}
- 성격: {characteristics}
- 영역명: {domain}
- 핵심 아이디어: {key_ideas}
- 내용체계: {content_elements}
- 성취기준: {standards}
- 교수학습 방법: {teaching_methods}
- 평가계획: {assessment_plan}

활동명: {data.get('activity_name')}
요구사항: {data.get('requirements')}

각 차시는 다음 사항을 고려하여 작성:
1. 명확한 학습주제 설정
2. 구체적이고 실천 가능한 학습내용 기술
3. 실제 수업에 필요한 교수학습자료 명시
4. 이전 차시와의 연계성 고려
5. 단계적 구성
6. 아래 예시를 참고하여 작성해주세요.
(예시)
학습주제: 질문 약속 만들기
학습내용: 질문을 할 때 지켜야 할 약속 만들기
         수업 중 질문, 일상 속 질문 속에서 갖추어야 할 예절 알기
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
                max_tokens=2000
            )
            response = chat(messages)
            content = response.content.strip().replace('```json', '').replace('```', '').strip()
            parsed = json.loads(content)
            lesson_plans = parsed.get("lesson_plans", [])
            for i, plan in enumerate(lesson_plans, start=start+1):
                plan["lesson_number"] = str(i)
            all_lesson_plans.extend(lesson_plans)
            time.sleep(1)
        except json.JSONDecodeError as e:
            st.error(f"{start+1}~{end}차시 생성 중 JSON 파싱 오류: {e}")
            continue
        except Exception as e:
            st.error(f"{start+1}~{end}차시 생성 중 오류: {e}")
            continue
    progress_bar.progress(100)
    return all_lesson_plans

def show_step_6(vector_store):
    total_hours = st.session_state.data.get('total_hours', 30)
    st.markdown(f"<div class='step-header'><h3>6단계: 차시별 지도계획 ({total_hours}차시)</h3></div>", unsafe_allow_html=True)
    if 'generated_step_5' not in st.session_state:
        with st.form("lesson_plans_form"):
            st.info(f"{total_hours}차시 계획 생성 중...")
            submit_button = st.form_submit_button(f"{total_hours}차시 생성 및 다음 단계로", use_container_width=True)
        if submit_button:
            with st.spinner("차시별 계획 생성 중..."):
                chunk_size = 10
                all_plans = generate_lesson_plans_in_chunks(total_hours, st.session_state.data, chunk_size, vector_store)
                if all_plans:
                    st.session_state.data['lesson_plans'] = all_plans
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

# 7단계: 최종 계획서 검토 및 Excel 다운로드
def show_final_review(vector_store):
    st.title("최종 계획서 검토")
    try:
        data = st.session_state.data
        tabs = st.tabs(["기본정보", "내용체계계", "성취기준", "교수학습 및 평가", "차시별계획"])

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
                "개요": data.get('overview',''),
                "성격": data.get('characteristics','')
            }
            for k,v in basic_info.items():
                st.markdown(f"**{k}**: {v}")
            st.button("기본정보 수정하기", key="edit_basic_info", on_click=lambda: set_step(1), use_container_width=True)

        with tabs[1]:
            st.markdown("### 영역/핵심아이디어/내용요소")
            domain = data.get("domain","")
            key_ideas = data.get("key_ideas",[])
            content_elements = data.get("content_elements",{})

            st.markdown("#### 영역명")
            st.write(domain)
            st.markdown("#### 핵심 아이디어")
            for idea in key_ideas:
                st.write(f"- {idea}")
            st.markdown("#### 내용 요소")
            st.write("**지식·이해**")
            for item in content_elements.get("knowledge_and_understanding",[]):
                st.write(f"- {item}")
            st.write("**과정·기능**")
            for item in content_elements.get("process_and_skills",[]):
                st.write(f"- {item}")
            st.write("**가치·태도**")
            for item in content_elements.get("values_and_attitudes",[]):
                st.write(f"- {item}")

            st.button("내용체계 수정하기", key="edit_goals_content", on_click=lambda: set_step(2), use_container_width=True)

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
            st.button("성취기준 수정하기", key="edit_standards", on_click=lambda: set_step(3), use_container_width=True)

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
                crit = ap.get("criteria","")
                st.markdown(f"**{code}** - {desc}")
                st.write(f"- 평가요소: {elem}")
                st.write(f"- 평가방법: {meth}")
                st.write(f"- 평가기준: {crit}")
                st.markdown("---")
            st.button("교수학습 및 평가 수정하기", key="edit_teaching_assessment", on_click=lambda: set_step(4), use_container_width=True)

        with tabs[4]:
            st.markdown("### 차시별 계획")
            lesson_plans_df = pd.DataFrame(data.get("lesson_plans", []))
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
            st.button("차시별 계획 수정하기", key="edit_lesson_plans", on_click=lambda: set_step(5), use_container_width=True)

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
                '개요': data.get('overview', ''),
                '성격': data.get('characteristics', '')
            }])
            basic_info.T.to_excel(writer, sheet_name='기본정보', header=['내용'])
            worksheet = writer.sheets['기본정보']
            for idx, col in enumerate(basic_info.T.index, 1):
                worksheet.set_column(idx, idx, 30, content_format)

        if "내용체계" in selected_sheets:
            content_sets = data.get("content_sets", [])
            if not content_sets:
                # 내용체계가 비어 있으면 빈 시트 생성
                df_empty = pd.DataFrame([{"구분": "내용체계 없음", "내용": ""}])
                df_empty.to_excel(writer, sheet_name='내용체계', index=False)
                worksheet = writer.sheets['내용체계']
                worksheet.set_column('A:A', 20, content_format)
                worksheet.set_column('B:B', 80, content_format)
            else:
                # 여러 세트(4세트 등)를 전부 rows에 담아서 Excel로
                rows = []
                for idx, cset in enumerate(content_sets, start=1):
                    domain = cset.get("domain", "")
                    key_ideas = cset.get("key_ideas", [])
                    ce = cset.get("content_elements", {})

                    # 영역명
                    rows.append({
                        "구분": f"영역명 (세트{idx})",
                        "내용": domain
                    })

                    # 핵심 아이디어
                    for idea in key_ideas:
                        rows.append({
                            "구분": f"핵심 아이디어 (세트{idx})",
                            "내용": idea
                        })

                    # 지식·이해
                    for item in ce.get("knowledge_and_understanding", []):
                        rows.append({
                            "구분": f"지식·이해 (세트{idx})",
                            "내용": item
                        })

                    # 과정·기능
                    for item in ce.get("process_and_skills", []):
                        rows.append({
                            "구분": f"과정·기능 (세트{idx})",
                            "내용": item
                        })

                    # 가치·태도
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
            # 교수학습방법: teaching_methods_text
            # 평가계획: assessment_plan (code, description, element, method, criteria)

            sheet_rows = []
            # 1) teaching_methods_text
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
                            "평가방법": line.strip(),
                            "평가기준": ""
                        })

            # 2) assessment_plan
            for ap in data.get('assessment_plan', []):
                sheet_rows.append({
                    "유형": "평가계획",
                    "코드": ap.get("code",""),
                    "성취기준": ap.get("description",""),
                    "평가요소": ap.get("element",""),
                    "평가방법": ap.get("method",""),
                    "평가기준": ap.get("criteria","")
                })

            df_methods = pd.DataFrame(sheet_rows)
            df_methods.to_excel(writer, sheet_name='교수학습및평가', index=False)
            worksheet = writer.sheets['교수학습및평가']
            worksheet.set_column('A:A', 14, content_format) # 유형
            worksheet.set_column('B:B', 14, content_format) # 코드
            worksheet.set_column('C:C', 30, content_format) # 성취기준
            worksheet.set_column('D:D', 30, content_format) # 평가요소
            worksheet.set_column('E:E', 30, content_format) # 평가방법
            worksheet.set_column('F:F', 30, content_format) # 평가기준

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

# 10. 자율시간 챗봇 기능 (사이드바 챗봇)
def show_chatbot(vector_store):
    st.sidebar.markdown("## 학교자율시간 챗봇")

    # 추천 질문
    st.sidebar.markdown("**추천 질문:**")
    recommended_questions = [
        "학교자율시간의 교육적 의의는 무엇인가요?",
        "자율시간 운영에 필요한 자료는 무엇인가요?",
        "자율시간 수업의 효과적인 진행 방법은?"
    ]
    for q in recommended_questions:
        if st.sidebar.button(q, key=f"rec_{q}"):
            st.session_state.chat_input = q

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
                max_tokens=512
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

# 11. 메인 함수
def main():
    try:
        set_page_config()
        if 'data' not in st.session_state:
            st.session_state.data = {}
        if 'step' not in st.session_state:
            st.session_state.step = 1
        st.title("2022 개정 교육과정 학교자율시간 계획서 생성기")

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
