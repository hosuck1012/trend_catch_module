# Trend Catch V2 개발 보고서

### — 기존 버전 대비 구조 및 기술 개선 —

> **핵심 요약**  
> 기존 Trend Catch가 뉴스와 YouTube에서 **트렌드 키워드를 찾는 시스템**에 가까웠다면, V2는 사회·문화 트렌드를 **실제 여행 콘텐츠 기회와 연관 여행지로 연결하는 근거 기반 고정밀 AI Pipeline**으로 발전했다. 이 변화의 핵심은 AI에게 더 많은 판단을 맡긴 것이 아니다. Gemini 이전에 `KeywordContext + NER + Rule + Local E5 + Ranking + Evidence Gate`를 강화하여 AI가 판단해야 할 후보 자체를 극소수로 줄였고, 최근의 Related Destination Expansion을 통해 질문의 범위를 “이 트렌드가 여행과 관련 있는가?”에서 “어느 여행지와 어떤 근거로 연결할 수 있는가?”까지 확장했다.

본 보고서에서는 Trend Catch V2 이전의 구현 상태를 편의상 **‘기존 버전(V1)’**으로 표기한다.

## V1 → V2 핵심 변화 요약

| 항목 | 기존 버전 | V2 | 개선 효과 |
|---|---|---|---|
| 입력 데이터 | 뉴스 RSS·YouTube 중심 | 뉴스·YouTube + Google Year in Search 2025 Korea static seed | 단기 이슈와 연간 검색 문화를 함께 탐색 |
| Keyword | 토큰·단어 중심 | Keyword Quality + Multi-token Phrase Recall + protected phrase | 긴 작품명·행사명 보존과 일반어 제거를 동시에 달성 |
| Context | 문서와 키워드 출현 연결 | 이전·일치·다음 문장을 가진 `KeywordContext` | 키워드가 등장한 이유와 주변 근거 보존 |
| NER | GLiNER·사전·규칙 초기 구조 | chunk·offset·protected entity·처리 상태/hash를 갖춘 NER V2 | 긴 문서와 복합 객체의 회수율 및 재처리 안정성 향상 |
| 여행 판별 | 키워드와 AI 설명 중심 | Rule Prefilter + 14개 Travel Taxonomy | 명시적 긍정·부정 근거로 비여행 후보 조기 제거 |
| Semantic | 생성형 해석 의존 가능성 | CPU 기반 `intfloat/multilingual-e5-small`의 positive/negative anchor 비교 | 외부 호출 없이 의미 유사도와 부정 의미 우세를 판별 |
| Ranking | 트렌드 점수 중심 | Trend 30% + Context 20% + Travel 30% + Evidence 20% | 인기와 실제 여행 전환 가능성을 분리해 종합 평가 |
| Evidence | 일반적인 분석 근거 | `PASS` / `NEEDS_EVIDENCE` / `REJECT` | 높은 점수만으로 장소를 확정하는 오류 방지 |
| 연관 여행지 | 모델 추천 가능성 | 공식 URL 기반 Related Destination Expansion | 테마형 여행지와 연결 근거를 함께 제공 |
| Gemini | 선택 키워드 설명·추천 | 근거를 통과한 소수 후보의 최종 판정과 표현 | 호출량·환각 위험 감소, 서버 후검증 강화 |
| 최종 결과 | 상위 트렌드 키워드 | 근거·여행 카테고리·검증 상태를 가진 콘텐츠 기회 | 발견에서 여행 기회 해석·검증·연결로 확장 |

## 목차

1. 프로젝트 개요 및 V2 개발 배경
2. 기존 버전의 문제점
3. V1 → V2 전체 Pipeline 비교
4. 뉴스/YouTube 기반 입력 구조
5. Google Year in Search 2025 Korea Seed
6. Keyword Quality 개선
7. Multi-token Phrase Recall
8. Protected Phrase / suffix / normalized dedup
9. KeywordContext
10. Cursor Pagination / Materialization 개선
11. GLiNER 기반 NER V2
12. Entity Linking
13. Rule-based Travel Prefilter
14. 14개 Travel Taxonomy
15. Local Semantic Filter
16. Semantic Precision 개선
17. High Precision Ranking
18. Evidence Gate
19. Related Destination Expansion
20. Gemini 역할 변화
21. Hallucination 방지
22. Cache / Hash / Idempotency
23. Dashboard 개선
24. 최종 실제 실행 결과
25. V1 vs V2 비교표
26. 한계점 및 향후 개선 방향
27. 결론

---

## 1. 프로젝트 개요 및 V2 개발 배경

Trend Catch Module은 YouTube 공식 API와 허용된 뉴스 RSS에서 문서를 수집하고, 문서에 반복적으로 등장하거나 빠르게 증가하는 키워드를 주간 단위로 계산하는 시스템이다. 초기 버전은 최근 7일과 이전 7일의 언급량, 증가율, 지속성, 출처 다양성, 검색 관심도 등을 조합해 `weekly_trend`, `watchlist`, `stable`, `insufficient_data`를 구분했다. 이 구조는 “지금 무엇이 뜨는가?”를 설명하는 데에는 유효했다.

그러나 Travelen과 같은 여행 서비스에 활용하려면 한 단계가 더 필요했다. 검색량이 높거나 뉴스에 많이 등장했다는 이유만으로는 실제 여행 콘텐츠가 되지 않는다. 예를 들어 인물명, 주가 기사, 사건·사고, 일반 서술어도 트렌드 점수는 높을 수 있지만 방문 장소, 행사 일정, 체험 행동으로 연결되지 않을 수 있다. 반대로 `폭싹 속았수다`, `설봉산 별빛축제`, `RESTOPIA`처럼 여행성이 제목 한 단어에 직접 드러나지 않는 고유명사는 단순 단어 기반 필터에서 놓치기 쉽다.

V2의 개발 목표는 다음 세 질문을 순서대로 해결하는 것이었다.

| 단계 | 질문 | 담당 계층 |
|---|---|---|
| 탐지 | 지금 사회·문화적으로 주목받는 것은 무엇인가? | 뉴스/YouTube, 주간 추세, Year in Search seed |
| 판별 | 해당 트렌드는 실제 여행 행동으로 전환 가능한가? | Context, NER, Rule, Local E5, Semantic Precision |
| 검증·확장 | 어느 장소와 어떤 근거로 연결할 수 있는가? | Ranking, Evidence Gate, Related Destination Expansion, 제한된 Gemini |

V2의 중심은 생성형 AI가 아니라 사전 검증 계층이다. 통계 점수, 문맥, 객체, 지역 근거와 로컬 의미 유사도를 먼저 계산하고, 충분히 강한 소수 후보만 Gemini에 전달한다. 비용 절감은 부수 효과이며, 더 중요한 효과는 **재현 가능한 판단 경로와 환각 억제**이다.

## 2. 기존 버전의 문제점

### 2.1 단어 단위 추출이 고유명사를 분해하는 문제

초기 키워드 추출은 토큰과 빈도를 중심으로 동작했다. 이 방식에서는 `설봉산 별빛축제`가 `설봉산`, `별빛`, `축제`로 분해되거나, `2026 국악관현악축제`의 숫자와 복합 명칭이 사라질 수 있었다. `폭싹 속았수다 촬영지`도 작품명과 여행 수정어의 결합을 하나의 기회 단위로 보존해야 하지만 단순 토큰화에서는 안정적으로 유지되지 않았다.

원인은 품질 필터가 “짧고 일반적인 단어를 제거”하는 데 최적화되어 있었고, “길지만 구체적인 복합명사를 보존”하는 신호가 부족했기 때문이다. V2는 단순히 허용 기준을 낮추지 않았다. 그렇게 하면 불량 구문도 함께 증가하기 때문이다. 대신 제목·인용부호·NER·접미사 사전·정규화 결과를 결합해 구체적인 구문에만 우선권을 부여했다.

### 2.2 키워드가 등장한 이유를 잃는 문제

V1의 `KeywordOccurrence`는 문서와 키워드의 연결은 보존했지만, 키워드가 어느 문장에서 어떤 주변 문장과 함께 등장했는지를 독립된 근거로 다루지 않았다. `제주`가 관광 기사에 나온 것인지, 날씨나 사건 기사에 나온 것인지, `펜타포트`가 공연 개최 문맥인지 단순 회고 문맥인지 구분하기 어려웠다.

이 문제는 이후 모든 단계에 영향을 준다. NER가 장소를 찾더라도 그 장소가 키워드와 같은 문맥에 있는지 알기 어렵고, 임베딩 모델은 긴 문서 전체의 잡음에 끌리며, Gemini에는 필요 이상의 원문을 보내게 된다. V2는 키워드 주변 문장을 `KeywordContext`로 물질화하여 문맥을 독립적인 계산·검증 단위로 만들었다.

### 2.3 여행 연관성과 실제 여행 가능성을 혼동하는 문제

`방문`, `지역`, `문화`, `행사` 같은 단어가 있다는 이유만으로 여행 후보를 만들면 false positive가 쉽게 발생한다. “기업 행사 방문”, “지역 주가”, “문화적 논쟁”도 표면적으로는 여행 anchor와 유사하다. 반대로 작품명이나 밈은 직접적인 여행 단어가 없어도 촬영지, 팝업, 지역 체험으로 전환될 수 있다.

V2는 이 문제를 Rule과 Semantic 중 하나로 해결하지 않고 계층화했다. Rule은 명시적 긍정·부정 신호와 객체 유형으로 넓게 거르고, Local E5는 긍정·부정 의미 anchor와의 거리를 비교한다. 이후 Semantic Precision이 주제 구체성, 객체 정렬, 문맥 근접성, 카테고리 근거를 다시 확인한다.

### 2.4 생성형 AI가 근거 밖의 장소를 보완할 위험

기존 AI 분석은 이미 저장된 근거를 해석하도록 설계되어 있었지만, 여행 추천이라는 과업 자체가 모델의 사전 지식을 끌어내기 쉽다. 작품명을 보고 유명 촬영지를 떠올리거나, 지역만 보고 세부 주소를 만들어낼 수 있다. 실제로 V2의 목표는 “그럴듯한 추천”이 아니라 “입력 근거로 검증 가능한 추천”이다.

따라서 V2는 Gemini가 장소를 자유 생성하지 못하도록 허용 장소 목록과 evidence ID를 제공하고, 출력 후 다시 서버에서 교정한다. 근거에 없는 장소는 `추가 검증 필요`로 치환하고, ACCEPT를 REVIEW로 낮춘다.

## 3. V1 → V2 전체 Pipeline 비교

```text
V1
YouTube / Newsis RSS
  → SourceDocument
  → 토큰 중심 KeywordOccurrence
  → 최근 7일 vs 이전 7일 WeeklyTrend
  → 검색 관심도 보정
  → 상위 키워드 Dashboard
  → 사용자가 선택한 키워드에 한해 AI 설명

V2
YouTube / Newsis RSS / Google Year in Search static seed
  → SourceDocument + KeywordOccurrence(v2)
  → Keyword Quality + Multi-token Phrase Recall
  → WeeklyTrend / watchlist
  → KeywordContext(이전·일치·다음 문장, hash)
  → GLiNER + 사전 + 규칙 NER V2
  → TrendEntityLink
  → Rule-based Travel Prefilter(14 taxonomy)
  → Local multilingual-E5 Semantic Filter
  → Semantic Precision(구체성·정렬·근접성·카테고리 근거)
  → High Precision Ranking + Evidence Gate
  → cluster 대표 및 Gemini budget 적용
  → 제한된 Gemini 최종 판정
  → Related Destination Expansion + 공식 출처 노출
```

가장 큰 차이는 출력 단위다. V1의 핵심 출력은 “점수가 높은 키워드”였지만 V2의 핵심 출력은 “근거, 여행 카테고리, 여행 전환 점수, 장소 검증 상태를 가진 여행 콘텐츠 기회”다.

## 4. 뉴스/YouTube 기반 입력 구조

V2도 V1의 수집 원칙을 유지한다. YouTube는 공식 YouTube Data API의 인기 동영상 정보를 사용하며 제목, 설명, 게시 시각, 조회·좋아요·댓글 수와 공식 영상 URL을 `SourceDocument`에 저장한다. 같은 `source + source_id`는 고유 제약으로 중복 저장되지 않는다.

뉴스는 임의 본문 크롤링이 아니라 코드에 허용된 뉴시스 RSS만 사용한다. RSS의 제목, 요약, 게시 시각, 원문 URL을 저장하고 GUID 또는 URL을 정규화·해시하여 source ID를 만든다. 동일 실행 안에서는 source ID와 URL을 모두 중복 검사하고, DB에서도 고유 제약을 적용한다.

| 입력 | 수집 방식 | 주요 필드 | 설계 이유 |
|---|---|---|---|
| YouTube | 공식 Data API | 제목, 설명, 게시일, 반응 지표, URL | 인기 콘텐츠와 사회·문화 반응 포착 |
| Newsis RSS | 허용 목록 RSS | 제목, 요약, 게시일, URL | 사건·문화·지역 행사에 대한 구조화된 공개 입력 |
| Google Year in Search | 코드에 고정된 static seed | 카테고리, 순위, 키워드, 공식 URL | 연간 검색 관심 키워드를 재현 가능하게 보강 |

두 외부 입력은 형식이 다르지만 모두 `SourceDocument → KeywordOccurrence`로 통일된다. 덕분에 이후 문맥, NER, 트렌드 계산, 여행 후보화 단계가 출처별 특수 로직에 종속되지 않는다.

## 5. Google Year in Search 2025 Korea Seed 추가

### 5.1 문제와 설계 선택

뉴스와 YouTube만으로는 연간 검색 문화의 대표 항목이 수집 시점에 따라 빠질 수 있다. 이를 보완하기 위해 커밋 `eed1049`에서 Google Year in Search 2025 Korea seed를 추가했다. 현재 코드에는 **17개 카테고리, 170개 키워드**가 명시되어 있다. 각 카테고리는 10개 순위 항목을 가지며 뉴스, 인물, 뜻 검색, 방법, AI Tools, 영화, 드라마/시리즈, K-POP 노래, K-POP 댄스, 스포츠 매치, 게임, 레시피, 여행지, 패션, 밈, 챌린지, 바이럴 간식으로 구성된다.

Google Trends 웹페이지 자동 크롤링이나 `pytrends`를 사용하지 않은 이유는 세 가지다.

1. Year in Search는 연간 공식 목록이며 매 실행마다 동적으로 긁을 필요가 없다.
2. 비공식 자동 크롤링은 페이지 구조 변경, 이용 정책, 재현성 문제를 만든다.
3. static seed는 어떤 키워드가 언제, 어떤 버전으로 들어왔는지 코드 리뷰와 테스트로 확인할 수 있다.

### 5.2 SourceDocument + KeywordOccurrence 연결

각 키워드는 최근 주간 구간 안의 0일, 2일, 4일 offset에 대응하는 세 개의 `SourceDocument`로 표현된다. 따라서 완전 신규 DB라면 최대 510개의 문서와 510개의 occurrence가 만들어진다. 문서에는 공식 Year in Search URL, 카테고리와 순위가 들어가며 occurrence에는 `keyword_quality_score=90`, `pipeline_version=v2`, `source=google_yis_2025_kr`가 기록된다.

```text
for each of 170 official keywords:
    normalize(keyword)
    for day_offset in [0, 2, 4]:
        upsert SourceDocument(source, category, rank, date)
        insert KeywordOccurrence if document+normalized_keyword is absent
```

source ID에는 `연도-국가:카테고리:순위:날짜`가 포함되어 반복 실행해도 동일 문서를 재생성하지 않는다. 관련 테스트는 seed API를 두 번 호출한 뒤 두 번째 `inserted_occurrences`가 0인지 확인한다.

### 5.3 watchlist 처리

Year in Search seed는 세 날짜에만 인위적으로 배치된 공식 목록이므로 일반적인 주간 증가율을 계산할 이전 주 데이터가 없다. 이 때문에 계산 결과가 `insufficient_data`에 머물면 공식 seed의 존재 목적이 사라진다. 서비스는 해당 source에서 들어온 공식 키워드 중 `insufficient_data`인 항목만 `watchlist`로 올린다. 독립적인 뉴스·YouTube 근거가 부족한 항목을 `weekly_trend`로 강제 승격하지 않는 이유는 연간 공식 검색 목록과 현재 주간 급등을 동일시하지 않기 위해서다.

테스트의 `상하이` 사례는 이 판단을 구체적으로 보여준다. seed 후 재계산하면 `status=watchlist`, `source_count=1`, `keyword_quality_score=90`이어야 한다. 즉 seed는 신뢰할 수 있는 탐색 후보를 추가하지만, 다중 출처나 현재 급등을 조작하지 않는다.

> **시점 주의:** Year in Search 추가 이후 Dashboard 수치는 기존 640개 문서 기반 V2 검증과 다른 시점의 결과다. 본 보고서는 두 결과를 합치거나 동일 모집단의 전후 비교로 해석하지 않는다.

## 6. Keyword Quality 개선

V1의 품질 필터는 URL 조각, 영문 전치사, 짧은 일반어, 서술어를 제거하는 데 집중했다. 실제 Dashboard에서도 `com`, `by`, `on`, `to`, `the`, `있습니다`, `증가`, `방문` 같은 불량 키워드를 제거하는 효과가 있었다. 그러나 강한 제거 규칙은 복합 고유명사 내부의 일반어까지 손상시킬 수 있었다.

V2는 후보를 단순 통과/탈락으로 보지 않고 후보 유형, 추출기, 제목 등장 횟수, NER 유형과 confidence, 구문 구체성 신호를 품질 점수에 반영한다. `LOCATION`, `PLACE`, `EVENT`, `FOOD`, `CONTENT_TITLE` 같은 여행에 유용한 객체 근거가 있으면 가점을 주되, `PERSON`, `BRAND`, `MEME`는 문맥 없이 자동 승인하지 않는다.

문제 해결 방식은 “불용어를 줄이는 것”이 아니라 “전체 구문을 살리고 구성 단어는 계속 거절하는 것”이다. 테스트는 `설봉산 별빛축제` 전체는 승인하면서 `별빛`, `축제` 같은 일반 구성 요소가 독립 키워드로 남지 않는지 확인한다. 또한 검색 관심도 값이 없을 때 점수를 0으로 간주하지 않고 사용 가능한 구성 요소의 가중치를 재정규화해, “미관측”과 실제 점수 50을 구분한다.

## 7. Multi-token Phrase Recall

커밋 `8ad254a`와 `5312d34`는 다중 토큰 회수율과 rebuild 시 구문 보존을 개선했다. `noun_phrases`는 연속 명사에서 2~4개 토큰 조합을 만들고, 한국어 명사 조합은 띄어쓰기 형태와 붙여쓰기 형태를 함께 생성한다. `specific_phrases`는 최대 6개 토큰을 보되, 등록된 주제 접미사로 끝나는 구체적 구문만 선택한다.

대표 gold phrase 테스트에는 다음 실제 사례가 포함된다.

| 유형 | 회수 대상 사례 | 기대 효과 |
|---|---|---|
| 작품·촬영지 | `폭싹 속았수다`, `폭싹 속았수다 촬영지` | 작품명과 여행 의도 결합 보존 |
| 지역 행사 | `설봉산 별빛축제`, `홍천강 별빛음악 맥주축제` | 긴 행사명을 하나의 후보로 유지 |
| 대형 행사 | `펜타포트 락 페스티벌`, `경남고성공룡세계엑스포` | EVENT 후보 회수 |
| 문화 행사 | `부산국제불교박람회`, `2026 국악관현악축제` | 숫자·복합명사 허용 |
| 밈·팝업 | `RESTOPIA`, `RESTOPIA 팝업`, `한강 밤핑` | 영문 고유명과 신조어 보존 |

테스트는 위 gold phrase가 생성되고 `quality_score >= 45`로 승인되는지 확인한다. 동시에 `가격`, `친절`, `증가`, `방문`, `축제`, `전시`, `공연`, `여행`처럼 단독으로는 일반적인 표현을 계속 거절한다. 이는 recall을 높이면서 precision을 잃지 않으려는 설계다.

## 8. Protected Phrase / suffix / normalized dedup

### 8.1 Protected Phrase

`data/protected_phrases.json`에는 프로젝트에서 반복적으로 검증한 작품명·행사명·밈이 등록되어 있다. 이 목록만 사용하는 것은 아니다. 인용부호 안의 텍스트가 `영화`, `드라마`, `축제`, `팝업`, `촬영지` 같은 제목 cue와 인접하면 구조적으로 protected phrase가 될 수 있다. 반대로 뉴스 기사 속 일반 인용문은 제목 cue가 없으면 보호하지 않는다.

예를 들어 `신작 '바람이 분다' 촬영지 공개`에서는 `바람이 분다`를 보호하지만, 일반 발언인 `'가격 안정'`은 자동으로 작품명이 되지 않는다. `뱅크시: 스틸 히어`처럼 콜론을 포함한 제목도 정규화 후 하나의 작품명으로 유지한다.

### 8.2 suffix 사전

접미사 사전은 EVENT, TRAVEL, TREND_MODIFIER 세 그룹과 총 24개 normalized suffix를 가진다. `축제`, `페스티벌`, `박람회`, `엑스포`, `콘서트`, `촬영지`, `팝업`, `여행`, `챌린지` 등이 포함된다. 접미사만 단독으로 등장하면 승인하지 않고, 앞에 구체적인 주제가 붙을 때만 `specific_phrase` 신호를 준다. `에이전시`가 `전시` 접미사로 오인되는 예외도 명시적으로 방지한다.

### 8.3 normalized dedup

`부산 불꽃 축제`, `부산불꽃축제`, `부산  불꽃축제`는 표시 문자열은 달라도 동일 normalized keyword로 합쳐진다. 여러 추출기에서 동일 후보가 나오면 단순히 먼저 나온 값을 채택하지 않고 protected/specific/NER 근거를 병합한다. 그 결과 긴 구체 구문이 일반 토큰 후보에 밀리지 않으며, rebuild 뒤에도 같은 normalized key로 `KeywordOccurrence`와 연결된다.

## 9. KeywordContext

V2는 키워드 출현 주변을 `KeywordContext` 테이블에 물질화한다. 주요 필드는 다음과 같다.

| 필드 | 의미 | 이후 활용 |
|---|---|---|
| `previous_sentence` | 일치 문장 직전 문장 | 배경·원인 보강 |
| `matched_sentence` | 키워드가 실제 등장한 문장 | 핵심 직접 근거 |
| `next_sentence` | 일치 문장 다음 문장 | 결과·방문 행동 보강 |
| `combined_context` | 위 세 문장을 결합한 제한된 문맥 | Rule, E5, Ranking, Gemini 입력 |
| `context_hash` | 문맥의 SHA-256 식별값 | 중복 방지·변경 감지·캐시 무효화 |

문장 분리기는 제목과 본문을 처리하고, 정규화된 키워드가 붙여쓰기나 표기 차이 속에서도 어느 원문 구간과 대응하는지 위치를 복원한다. 동일 문서에서 키워드가 여러 번 등장하면 `occurrence_index`를 유지한다.

DB 고유 제약은 `(document_id, normalized_keyword, context_hash)`다. 같은 문맥은 반복 실행해도 중복 생성되지 않는다. 문서 내용이 바뀌어 hash가 달라지면 새 문맥으로 인식되고, 이전 문맥을 무작정 덮어써 이미 생성된 하위 근거를 잘못 연결하지 않는다.

## 10. Cursor Pagination / Materialization 개선

초기 V2 구현에서 큰 데이터셋을 한 번에 처리하면 일부 배치만 처리되거나, 재시작 시 첫 페이지부터 반복하는 문제가 생길 수 있었다. `a22305d`, `8963ecc`는 context와 rule 결과의 완전한 materialization을 목표로 cursor 기반 처리를 강화했다.

처리 방식은 offset이 아니라 증가하는 row ID를 cursor로 사용한다.

```text
cursor = after_id
while True:
    page, next_cursor, has_more = fetch(id > cursor, limit=batch_size)
    materialize or upsert(page)
    if not has_more: break
    assert next_cursor > cursor
    cursor = next_cursor
```

이 방식은 중간에 실패해도 마지막 ID 이후부터 재개할 수 있고, 앞선 행의 삽입·삭제로 offset이 흔들리는 문제를 줄인다. 서비스는 `next_cursor`가 진행하지 않으면 예외를 발생시켜 무한 루프를 방지한다. 또한 eligible row 수와 materialized row 수를 비교해 누락된 rule 결과를 계산한다.

Context materialization은 페이지 단위로 기존 `(document, keyword, hash)` 키를 미리 조회하고, 실제 신규 row만 insert한다. Rule 후보는 `rule_input_hash`가 같으면 재계산을 건너뛰며, 바뀌면 upsert한다. 이 개선은 단순 성능 최적화가 아니라, NER·Semantic·Ranking이 “전체 데이터가 처리되었다”는 전제 위에서 동작하게 만드는 정확성 개선이다.

## 11. GLiNER 기반 NER V2

V1에서도 multilingual GLiNER, 한국 지역 사전, 규칙 기반 보정이 있었지만 초기 운영 보고서의 640개 문서 중 EntityMention은 42개에 불과했다. 긴 문서 처리, 복합 행사명, 작품명·밈 보호, 재처리 상태 관리가 충분하지 않아 travel pipeline의 객체 근거가 희박했다.

커밋 `685250b`는 NER V2를 다음과 같이 개선했다.

1. 제목과 본문을 문장 단위 chunk로 나누고 GLiNER 안전 길이 상한을 적용한다.
2. chunk마다 원문 offset을 보존하고 결과를 문서 기준 위치로 다시 이동한다.
3. GLiNER, 지역 사전, 규칙 후보를 confidence·구체성·겹침 기준으로 병합한다.
4. `ner_protected_entities.json`으로 작품명, 행사명, 음식, 밈을 보호한다.
5. 모델·label version·pipeline version·문서 내용이 포함된 `input_hash`와 `EntityExtractionState`를 저장한다.
6. cursor, batch, dry-run, force 옵션으로 대량 재처리와 재시작을 안전하게 만든다.

NER gold case는 `경남고성공룡세계엑스포(EVENT)+고성(LOCATION)`, `설봉산 별빛축제(EVENT)+설봉공원(PLACE)`, `펜타포트 락 페스티벌(EVENT)+송도(LOCATION)`, `폭싹 속았수다(CONTENT_TITLE)` 등을 검증한다. `RESTOPIA`는 protected MEME로 유지된다. 반면 `가격 상승`, `정부 지원`, `실적 발표`, `주가 전망`, `법적 분쟁` 같은 표현은 여행 객체로 오인하지 않도록 negative case로 관리한다.

## 12. Entity Linking

EntityMention은 문서 안의 객체이고, TrendEntityLink는 주간 트렌드 키워드와 객체 사이의 집계 관계다. 링크는 키워드 occurrence가 있는 문서에서 객체를 모아 주차, normalized keyword, normalized entity, entity type 단위로 통합한다. 문서 수, 언급 수, 평균 confidence를 계산하고 대표 객체를 표시한다.

이 연결이 필요한 이유는 키워드와 장소가 문자열로 완전히 같지 않기 때문이다. `폭싹 속았수다`라는 키워드는 `제주`, 촬영 관련 장소, CONTENT_TITLE과 함께 등장할 수 있다. 여행 후보는 이 링크를 통해 “작품명 자체”와 “문서에서 함께 확인된 지역 객체”를 분리하면서도 관계를 유지한다.

링크는 `(keyword, week_start, normalized_entity, entity_type)` 고유 키를 사용해 반복 계산에 안전하다. V2 재물질화 과정에서 문서별 객체 coverage가 높아지면서 TrendEntityLink도 증가했고, Ranking의 entity stability·location evidence·event-location pair 계산에 사용될 근거가 크게 확장되었다.

## 13. Rule-based Travel Prefilter

Rule Prefilter의 목적은 최종 결정을 내리는 것이 아니라, 명백한 비여행 후보를 싸게 제거하고 의미 모델에 보낼 후보를 구조화하는 것이다. 입력은 KeywordContext, 주간 트렌드, 주요 entity, 긍정·부정 용어다.

기본 점수는 entity prior, positive context, trend, source diversity에서 얻고 negative context penalty를 뺀다. PLACE, LOCATION, EVENT는 높은 prior를 가지지만, PERSON과 BRAND는 낮은 prior를 갖는다. 주가·실적·투자 문맥의 BRAND, 재판·사고 문맥의 PERSON은 강하게 감점된다.

출력 상태는 `rejected`, `weak`, `review`, `strong`이다. 여기서 review/strong만 Semantic 단계의 핵심 대상이 된다. 단, Related Destination Expansion으로 공식 여행지 문맥이 연결되면 `RELATED_DESTINATION_VERIFIED`와 travel category를 받아 기존에는 약했던 콘텐츠 키워드도 검토 후보가 될 수 있다.

## 14. 14개 Travel Taxonomy

V2는 여행 연관성을 하나의 이진값으로 처리하지 않고 14개 카테고리로 분류한다.

| 카테고리 | 의미 | 대표 변환 |
|---|---|---|
| `FILM_LOCATION` | 영화 촬영지 | 로케이션 투어 |
| `DRAMA_LOCATION` | 드라마 촬영지 | 작품 성지순례 |
| `SHOW_LOCATION` | 예능 촬영지 | 방송 코스 재현 |
| `FESTIVAL` | 지역 축제 | 일정 기반 방문 |
| `CONCERT` | 콘서트·공연 | 공연 여행 |
| `EXHIBITION` | 전시·박람회 | 문화 관람 |
| `POPUP` | 팝업·기간 한정 공간 | 예약·대기형 방문 |
| `FOOD` | 음식·디저트·시장 | 미식 탐방 |
| `SPORTS_EVENT` | 경기·모터스포츠 | 직관·체험 |
| `LOCAL_CULTURE` | 지역 문화 | 로컬 체험 |
| `NATURE` | 산·섬·공원·해변 | 자연 탐방 |
| `LANDMARK` | 명소·랜드마크 | 장소 방문 |
| `REGIONAL_MEME` | 지역 기반 밈 | 밈 발생지·체험 콘텐츠 |
| `OTHER` | 근거가 불충분한 기타 | 낮은 전환성 유지 |

Taxonomy는 E5 positive anchor의 분류 단위이자 Travel Convertibility의 카테고리 가치, Evidence Gate의 장소 필요 조건, Related Destination Expansion의 연결 타입으로 함께 사용된다. 따라서 단순 표시용 label이 아니라 단계 간 계약이다.

## 15. Local Semantic Filter

커밋 `400943f`는 `intfloat/multilingual-e5-small` 기반 로컬 의미 필터를 추가했다. 기본 device는 `cpu`, batch size는 16이며 모델 인스턴스는 프로세스 내에서 캐시된다. 외부 임베딩 API를 호출하지 않으므로 문맥이 외부로 전송되지 않고 후보 수가 늘어도 호출 비용이 발생하지 않는다.

E5의 입력 규칙에 맞춰 anchor는 `query:`, 후보 문맥은 `passage:` prefix를 사용한다. `travel_semantic_anchors.json`에는 14개 여행 카테고리의 positive anchor 42개와 금융·법률·사고 등 5개 negative 그룹의 anchor 17개가 있다.

단순 positive similarity만 사용하지 않는 이유는 일반 뉴스 문장도 “지역”, “행사”, “방문”과 의미적으로 가깝기 때문이다. V2는 가장 가까운 positive와 negative를 모두 계산하고 그 차이인 margin을 본다. negative가 우세하면 `NEGATIVE_SEMANTIC_DOMINANT`, 차이가 너무 작으면 `LOW_SEMANTIC_MARGIN`을 부여한다. 모델명, anchor version, context hash, rule input hash, 후보 텍스트를 묶은 `embedding_input_hash`가 같으면 결과를 재사용한다.

## 16. Semantic Precision 개선

초기 Local E5는 recall은 확보했지만 일반 주제나 문서 먼 곳의 객체 때문에 false positive가 남았다. 커밋 `b580e48`과 `8b81d96`은 임베딩 점수 위에 해석 가능한 정밀도 계층을 추가했다.

### 16.1 Topic Specificity

`축제`, `전시`, `여행` 같은 단독 일반어와 `설봉산 별빛축제` 같은 구체 주제를 구분한다. protected phrase, specific phrase, NER 고유객체, 제목 반복 등의 품질 근거가 있어야 높은 구체성을 인정한다. 한 토큰 일반 주제, 형태가 깨진 조각, generic topic 사전에 있는 표현은 제한한다.

### 16.2 Entity Alignment

문서에 여행 객체가 있다는 사실만으로는 부족하다. normalized keyword와 entity text가 정확히 일치하는지(EXACT), 부분 포함인지(PARTIAL), 무관한지(UNALIGNED)를 구분한다. 예를 들어 문서 어딘가에 `서울`이 있어도 `RESTOPIA`와 같은 문맥에서 연결되지 않으면 강한 장소 근거가 아니다.

### 16.3 Context Locality

객체와 여행 신호가 `matched_sentence`, 인접 문장, 문서 전체 중 어디에 있는지 구분한다. 직접 일치 문장의 근거를 가장 강하게 보고, 인접 문장은 보조로 사용하며, 문서 먼 곳의 객체만 있는 경우 점수를 제한한다. 뉴시스 dateline(`[서울=뉴시스]`)처럼 본문 주제와 무관한 위치 표시는 별도 패턴으로 제거한다.

### 16.4 Category Evidence Matrix

카테고리마다 필요한 객체와 행동 근거가 다르다. FESTIVAL은 행사명과 개최/개막/열림 문맥을, DRAMA_LOCATION은 작품명과 촬영지·방문 문맥을, FOOD는 음식과 장소의 결합을 요구한다. `OTHER`나 객체 근거 없는 단순 E5 유사도는 높은 점수를 유지하지 못한다.

### 16.5 false positive 감소 과정

과거 V2 의미 보정 실행 기록에서는 분포가 다음과 같이 변했다. 이 수치는 640개 문서 기반 기존 V2 검증 시점의 기록이며, 현재 Year in Search 추가 이후 Dashboard Funnel과 동일 실험이 아니다.

| 상태 | Semantic OLD | Semantic NEW | 해석 |
|---|---:|---:|---|
| rejected | 183 | 392 | 애매한 후보를 명시적으로 제거 |
| weak | 334 | 547 | 즉시 여행 기회로 승격하지 않고 보류 |
| review | 128 | 8 | 사람이 볼 애매한 중간 구간 축소 |
| strong | 406 | 104 | 강한 후보 기준을 엄격하게 조정 |

같은 검증 기록에서 Top 50 false positive는 10건에서 0건으로 감소했다. 이는 strong 수를 늘린 결과가 아니라, strong을 406개에서 104개로 줄이면서 상위 정밀도를 높인 결과다. 관련 테스트는 대표 gold phrase가 reviewable 상태를 유지하는지와 기존 generic false-positive fixture가 reviewable로 남지 않는지를 함께 확인한다.

## 17. High Precision Ranking

Semantic을 통과했다고 모두 Gemini 후보가 되는 것은 아니다. Ranking은 동일 normalized keyword의 문맥·객체·semantic row를 모아 네 축으로 다시 점수화한다.

| 축 | 가중치 | 주요 근거 |
|---|---:|---|
| Trend Strength | 30% | final/trend/growth/acceleration/persistence, 출처·문서 수, 검색 관심도, watchlist 상태 |
| Context Clarity | 20% | matched sentence의 키워드 포함, 적정 길이, 인접 문장 일관성, 문서 간 반복, entity stability |
| Travel Convertibility | 30% | prefilter, taxonomy 가치, semantic score, 여행 객체 유형, 긍정 용어, 부정 패널티 |
| Evidence Confidence | 20% | 문서·출처 수, 문맥 반복, location/event pair, 객체 confidence, 검증 context |

```text
HighPrecision = Trend×0.30 + Context×0.20
              + Convertibility×0.30 + Evidence×0.20
```

후보는 문서 공유, 객체 공유, 키워드 유사도로 cluster를 만들고 가장 높은 점수 하나만 대표가 된다. `폭싹 속았수다`와 `폭싹 속았수다 촬영지`처럼 유사 근거를 공유하는 후보가 Gemini 예산을 중복 소비하는 것을 막기 위해서다. 대표 중 ranking status가 `gemini_candidate` 또는 `priority_candidate`이고 Evidence Gate가 PASS인 항목만 주간 budget 안에서 `gemini_eligible`이 된다.

## 18. Evidence Gate

Evidence Gate는 점수와 별개의 안전장치다. 높은 semantic score만으로 장소를 확정하지 못하게 한다.

| Gate | 조건의 의미 | 후속 처리 |
|---|---|---|
| `PASS` | 충분한 문서·출처·객체/장소 또는 검증 context가 있음 | 대표 cluster라면 Gemini 후보 가능 |
| `NEEDS_EVIDENCE` | 여행 가능성은 있으나 장소·출처·구체 근거가 부족 | REVIEW 유지, 외부 검증 문구 노출 |
| `REJECT` | 부정 의미 우세, 문맥 없음, OTHER, 낮은 rule+semantic | 최종 후보 제외 |

Evidence code는 결과를 설명 가능하게 만든다. 예: `MULTI_DOCUMENT_CONFIRMATION`, `MULTI_SOURCE_CONFIRMATION`, `LOCATION_EVIDENCE`, `EVENT_LOCATION_PAIR`, `CONTENT_TITLE_CONTEXT`, `NEGATIVE_SEMANTIC_DOMINANT`, `AMBIGUOUS_CONTEXT`, `RELATED_DESTINATION_CONTEXT`, `OFFICIAL_DESTINATION_SOURCE`.

중요한 점은 Related Destination이 있다고 자동 PASS가 되지 않는다는 것이다. 공식 연관 여행지는 제안 근거를 추가하지만, 원래 트렌드 문서 안에서 실제 촬영지나 개최지가 검증된 것과는 다르다. 따라서 F1 사례 테스트에서도 세 개 공식 연관 여행지가 연결된 결과는 `review + NEEDS_EVIDENCE`로 유지된다.

## 19. Related Destination Expansion

### 19.1 질문의 확장

V2 초기 단계가 **“이 트렌드가 여행과 관련 있는가?”**를 판단하는 데 초점을 맞췄다면, 최종 확장에서는 **“어느 여행지와 어떤 근거로 연결할 수 있는가?”**까지 판단 범위를 확장하였다. 커밋 `3881725`의 Related Destination Expansion은 이 질문에 공식 출처가 있는 연관 여행지로 답하기 위한 계층이다.

`data/travel_destination_expansion_catalog.json`은 버전이 있는 curated catalog다. 현재 구현된 첫 규칙은 Google Year in Search의 영화 키워드 `F1 더 무비`를 `MOTORSPORT` 테마 및 `SPORTS_EVENT` 여행 카테고리에 연결한다. source가 반드시 `google_yis_2025_kr`, category가 영화여야 하므로 같은 문자열이 임의의 YouTube 문서에 나왔다는 이유만으로 확장되지 않는다.

### 19.2 공식 정보 기반 여행지 제안

현재 catalog는 다음 세 장소를 제안한다.

| 연관 여행지 | 지역 | 제안 활동 | 근거 |
|---|---|---|---|
| 코리아 인터내셔널 서킷 | 전라남도 영암군 | 국내 모터스포츠 경기·서킷 행사 연계 | 대한자동차경주협회 자료 URL |
| 인제스피디움 | 강원특별자치도 인제군 | 서킷 택시·사파리·카트·박물관 | 인제스피디움 공식 소개 |
| BMW 드라이빙 센터 | 인천광역시 중구 | 예약형 트랙·자동차 문화 체험 | 공식 홈페이지 |

각 항목은 destination ID, 이름, 지역, 객체 유형, 활동, 방문 조건, 공식 URL, 출처 제목, match confidence를 가져야 한다. URL은 HTTP(S)인지 검증하고 entity type은 PLACE 또는 LOCATION으로 제한한다.

### 19.3 서비스·모델·API 연결

`related_destination_expansion_service`는 catalog를 읽어 `EntityContext`와 `TrendContextLink`로 물질화한다. page ID는 `travel-destination:{theme}:{destination}` 형태이며, catalog version을 revision으로 사용한다. API `POST /api/travel-opportunities/expand-destinations`는 dry-run과 실제 반영을 구분한다.

테스트에서는 dry-run 시 3개 context와 link가 예상되지만 DB row는 0인지, 실제 첫 실행은 각각 3개를 만들고 두 번째 실행은 0개 생성·3개 skip인지 확인한다. 이후 Prefilter는 `RELATED_DESTINATION_VERIFIED`, Semantic은 공식 destination context를 후보 문맥으로 사용하고, Ranking은 `RELATED_DESTINATION_CONTEXT`를 evidence code로 추가한다. 상세·목록·calibration API 모두 `related_destinations` 배열을 반환한다.

이 기능은 생성형 추천과 다르다. 모델이 임의로 장소를 떠올리는 대신, 사람이 검토한 공식 URL과 방문 조건이 있는 catalog가 확장 범위를 결정한다. 향후 catalog가 확대되더라도 같은 원칙을 유지해야 한다.

## 20. Gemini 역할 변화

V1의 Gemini는 상위 키워드의 상승 이유와 여행 연관성을 설명하는 보조 분석기였다. V2에서는 호출 전 조건과 호출 후 검증이 훨씬 엄격해졌다.

| 구분 | V1 | V2 |
|---|---|---|
| 호출 후보 | 사용자가 선택한 상위 키워드 | cluster 대표 + ranking status + Evidence PASS + 주간 budget |
| 입력 | 키워드와 기존 DB 근거 | 제한된 Evidence Package와 허용 ID/장소 목록 |
| 역할 | 설명·추천 보조 | 소수 고정밀 후보의 최종 accept/review/reject와 콘텐츠 표현 |
| 금지 | 통계 순위 변경 금지 | 순위 변경, 외부 기억, 웹 검색, 새 장소·source 생성 금지 |
| 출력 후 처리 | schema 검증 | evidence ref·장소 검증, decision 강등, partial 상태 저장 |

즉 V2는 AI에게 더 많은 판단을 맡긴 것이 아니다. `Context + NER + Rule + Local E5 + Ranking + Evidence Gate`를 강화하여 AI가 볼 후보를 극소수로 줄였다. 최신 Dashboard에서는 Raw 2,866개 중 Gemini Eligible은 1개로, 화면에 LLM 호출 감소율 99.97%가 표시된다.

## 21. Hallucination 방지

Gemini system instruction은 Evidence Package 안의 정보만 사실로 사용하고 외부 기억·웹 검색으로 장소를 추가하지 말라고 명시한다. 문서 안의 프롬프트 주입 가능성을 줄이기 위해 evidence를 `<untrusted_evidence>`로 감싸고 그 안의 지시는 따르지 않도록 한다.

서버는 모델 출력을 그대로 저장하지 않는다.

1. `evidence_refs`가 실제 package의 `CTX-*`, `DOC-*`, `ENTITY-*`, `CONTEXT-*`, `RANKING-V2` 중 하나인지 검사한다.
2. 추천 장소가 입력의 LOCATION/PLACE, TrendEntityLink, matched/manual context에 존재하는지 확인한다.
3. 허용되지 않은 장소는 콘텐츠 아이디어에서 `추가 검증 필요`로 바꾼다.
4. 검증되지 않은 장소가 있으면 `needs_external_verification=true`와 검색 제안을 만든다.
5. Evidence Gate가 NEEDS_EVIDENCE인데 모델이 ACCEPT하면 REVIEW로 강등한다.
6. 비여행·금융성 후보는 모델이 ACCEPT해도 REJECT로 교정하고 콘텐츠 아이디어를 제거한다.

`폭싹 속았수다` 실제 결과가 이를 보여준다. 입력 근거에서 `제주`는 확인되어 표시하지만 구체 촬영 장소와 주소는 확정하지 않는다. 화면에는 “상세 촬영 장소는 외부 검증이 필요”라는 경고와 검색 제안이 함께 표시된다.

## 22. Cache / Hash / Idempotency

V2는 재실행 가능성을 기능 요구사항으로 본다.

| 계층 | 식별·캐시 수단 | 효과 |
|---|---|---|
| SourceDocument | `(source, source_id)` unique | 수집·seed 중복 방지 |
| KeywordOccurrence | 문서+normalized keyword 조회 | seed/rebuild 중복 방지 |
| KeywordContext | document+keyword+context hash unique | 같은 근거의 중복 생성 방지 |
| NER | document/model/label/pipeline/content input hash | 변경 없는 문서 skip |
| Rule | context·trend·entity·version 기반 rule input hash | prefilter 재계산 최소화 |
| Semantic | model·anchor·context·rule 기반 embedding input hash | CPU embedding 캐시 |
| Ranking | force 여부와 저장 후보 upsert | 동일 주차 결과 갱신 |
| Gemini | model·prompt version·canonical evidence input hash | 같은 근거 재호출 방지 |
| Destination | catalog version+theme+destination page ID | 확장 idempotency |

Gemini는 dry-run에서 실제 호출 없이 예상 호출 수와 budget을 확인할 수 있다. 실제 실행도 주간 최대 후보 수에서 이미 사용한 call 수를 빼고, cache hit는 호출로 계산하지 않는다. 이 구조는 비용뿐 아니라 같은 입력에 같은 결과를 연결하는 감사 가능성을 높인다.

## 23. Dashboard 개선

V1 Dashboard는 상위 트렌드, 검색 관심도, 출처 분포, AI 분석, 키워드 상세, 파이프라인 상태를 제공했다. V2는 별도의 `여행 기회 V2` 화면에 Funnel, 최종 결과, High Precision 세부 점수, Evidence Gate, Gemini eligibility, cluster, context, Related Destination을 노출한다.

[그림 1. Trend Catch Dashboard 전체 화면]  
파일: `reports/screenshots/01_dashboard_overview.png`

그림 1의 최신 화면에는 이번 주 트렌드 13개, watchlist 554개, 평균 final score 27.81, AI 분석 완료 6개, 수집 문서 1,073개가 표시된다. 출처 분포에는 `google_yis_2025_kr`가 추가되어 static seed가 기존 뉴스·YouTube 입력과 같은 Dashboard 집계에 들어왔음을 확인할 수 있다. 검색 관심도 데이터가 없는 상태는 빈 차트 대신 안내 메시지로 구분된다.

[그림 2. V2 여행 기회 Funnel]  
파일: `reports/screenshots/02_v2_funnel.png`

그림 2는 최신 Funnel의 Raw 2,866, Quality 2,699, Rule 42, Semantic 34, High Precision 3, Gemini Eligible 1을 보여준다. 후보 감소가 한 번의 AI 판정이 아니라 여러 결정적·로컬 계층을 거쳐 이루어졌다는 점을 시각화한다.

[그림 3. 최종 여행 기회 후보 상세]  
파일: `reports/screenshots/03_final_candidate.png`

그림 3에는 `폭싹 속았수다`의 최종 점수 82, 신뢰도 80, REVIEW, 제주, context 5개와 entity 3개에 해당하는 evidence ref가 표시된다. High Precision Score는 82.73이며 Trend Strength 82.6, Context Clarity 87.0, Travel Convertibility 82.15, Evidence Confidence 79.51, Evidence Gate PASS, Gemini Eligible YES가 보인다.

[그림 4. 연관 여행지 제안 화면]
파일: `reports/screenshots/04_related_destinations.png`

관련 Dashboard 코드는 High Precision 카드 안에 공식 URL, 지역, 활동을 목록으로 표시하도록 구현되어 있다. 다만 현재 저장된 그림 4의 캡처 프레임에는 상단 Funnel과 `폭싹 속았수다` REVIEW 카드가 주로 보이고, 연관 여행지 목록 자체는 화면 아래 영역이라 이미지에서 직접 판독되지 않는다. 따라서 본 보고서의 여행지 세부 내용은 catalog·service·API 테스트를 근거로 기술했다.

[그림 5. Evidence 및 외부 검증 화면]
파일: `reports/screenshots/05_evidence_verification.png`

현재 그림 5에서도 REVIEW 카드의 입력 근거 ID, 외부 검증 경고, 확인 필요 문구와 검색 제안을 확인할 수 있다. High Precision 카드의 `Evidence Codes` 전체 목록은 캡처 하단 밖에 있지만, 그림 3에는 Evidence Gate PASS가 보이며 Dashboard 코드와 API schema는 Evidence Codes, 문서 수, 출처 수를 명시적으로 노출한다.

## 24. 최종 실제 실행 결과

### 24.1 기존 V2 검증 시점: SourceDocument 640개 기반

다음 값은 Year in Search seed 추가 전, 640개 SourceDocument를 대상으로 진행된 기존 V2 재물질화·정밀도 검증 시점의 기록이다. 저장소의 이전 보고서는 640개, 초기 EntityMention 42개, TrendEntityLink 182개를 확인한다. 이후 값은 V2 개발 과정의 검증 기록으로 제공되었으나 원시 실행 로그가 현재 reports에 보존되어 있지 않아 이번 문서 작업에서 재실행으로 재현하지는 않았다.

| 지표 | 개선 전/후 또는 결과 | 의미 |
|---|---:|---|
| SourceDocument | 640 | 동일 기존 검증 모집단 |
| EntityMention | 42 → 4,189 | chunk·protected entity·rule 보강 효과 |
| Entity 문서 coverage | 5/640 → 581/640 | 객체 근거가 거의 전 문서로 확대 |
| TrendEntityLink | 182 → 1,484 | 트렌드-객체 연결 근거 확대 |
| KeywordContext | 약 16,674 | 문장 단위 근거 materialization |
| Semantic OLD | rejected 183 / weak 334 / review 128 / strong 406 | 초기 local E5 후 분포 |
| Semantic NEW | rejected 392 / weak 547 / review 8 / strong 104 | precision 계층 적용 후 분포 |
| Top 50 False Positive | 10 → 0 | 상위 정밀도 개선 |
| Semantic ranking 대상 | distinct keyword 33 | ranking 입력 규모 |
| Gemini Eligible | 1 | AI 이전 후보 극소화 |
| 실제 Gemini 호출 | 1 | 무차별 대량 호출 방지 |

최종 후보는 `폭싹 속았수다`였다. 결과는 ACCEPT가 아니라 **REVIEW**, 최종 점수 82, confidence 80, 입력 근거로 확인된 destination은 제주였다. Context Evidence 5개와 Entity Evidence 3개를 참조했고, 상세 촬영지는 입력 근거에 없으므로 외부 검증 필요로 남겼다.

### 24.2 Year in Search 및 Destination Expansion 이후 최신 Dashboard 시점

최신 캡처는 별도 시점이다. Google Year in Search seed와 Related Destination Expansion 코드가 추가된 뒤의 현재 UI 상태이며, 위 640개 실험과 직접적인 전후 비교로 합산할 수 없다.

| Dashboard 지표 | 최신 캡처 값 |
|---|---:|
| 수집 문서 수 | 1,073 |
| 이번 주 트렌드 | 13 |
| watchlist | 554 |
| Raw | 2,866 |
| Quality | 2,699 |
| Rule | 42 |
| Semantic | 34 |
| High Precision | 3 |
| Gemini Eligible | 1 |
| LLM 호출 감소율 | 99.97% |
| 연율화 후보 추정 | 52.0 |

최신 화면에서도 확정된 최종 여행 콘텐츠 기회는 0개이며, `폭싹 속았수다`가 REVIEW로 표시된다. 이는 파이프라인이 결과를 억지로 ACCEPT하지 않고 근거 부족을 사용자에게 드러내는 설계가 실제 UI까지 이어졌음을 보여준다.

### 24.3 실제 사례별 의미

| 사례 | V2에서 검증하는 핵심 |
|---|---|
| 폭싹 속았수다 | CONTENT_TITLE 보존, 제주 entity 연결, 근거 밖 촬영지 생성 금지 |
| 폭싹 속았수다 촬영지 | 작품명+여행 modifier의 multi-token 보존 |
| 설봉산 별빛축제 | EVENT와 설봉공원 PLACE의 동시 회수 |
| 홍천강 별빛음악 맥주축제 | 긴 복합 행사명 phrase recall |
| 펜타포트 락 페스티벌 | EVENT+송도 LOCATION 및 개최 문맥 |
| 경남고성공룡세계엑스포 | 붙여쓴 대형 행사명과 고성 LOCATION |
| 부산국제불교박람회 | 문화 행사 전체 구문과 LOCAL_CULTURE 가능성 |
| 2026 국악관현악축제 | 숫자가 포함된 행사명 보존 |
| RESTOPIA | 영문 밈/브랜드성 표현을 protected MEME로 보존하되 문맥 없이 여행 확정 금지 |

## 25. V1 vs V2 비교표

| 관점 | 기존 Trend Catch | Trend Catch V2 |
|---|---|---|
| 목표 | 주간 트렌드 키워드 탐지 | 근거 기반 여행 콘텐츠 기회 발굴 |
| 입력 | 뉴스·YouTube | 뉴스·YouTube + 공식 Year in Search static seed |
| 키워드 단위 | 토큰·단어 중심 | protected/specific multi-token phrase 중심 |
| 품질 관리 | 불용어·일반어 제거 | 후보 유형·NER·제목·접미사·구체성 점수 |
| 문맥 | occurrence가 연결한 문서 | 이전/일치/다음 문장의 KeywordContext |
| NER | GLiNER+사전+규칙 초기형 | chunk, offset, protected entity, state/hash, batch |
| 객체 연결 | 제한적 TrendEntityLink | 문서 coverage 확대 및 ranking evidence 활용 |
| 여행 판별 | 키워드/AI 설명 중심 | Rule + 14 taxonomy + Local E5 + precision matrix |
| 의미 모델 | 없음 또는 생성형 해석 | CPU multilingual-e5-small positive/negative 비교 |
| 순위 | trend final score | 30/20/30/20 High Precision Score |
| 근거 상태 | 일반적인 분석 근거 | PASS / NEEDS_EVIDENCE / REJECT |
| AI 후보 | 사용자 선택 상위 키워드 | cluster 대표·Evidence PASS·budget 내 극소수 |
| AI 권한 | 설명·추천 | 입력 근거 내 최종 표현, 서버가 재검증·강등 |
| 장소 추천 | 모델 추천 가능성 | 허용 장소 검증 + curated official destination |
| 재실행 | 수집·계산 upsert | 각 단계 cursor/hash/cache/idempotency |
| Dashboard | 트렌드 중심 | Funnel, ranking 4축, evidence, final review, destination |

## 26. 한계점 및 향후 개선 방향

### 26.1 Destination catalog 범위

현재 `travel_destination_expansion_catalog.json`에는 `F1 더 무비`의 MOTORSPORT 규칙 한 개와 세 장소만 있다. 구조와 안전성은 검증됐지만 범용 destination recommender라고 보기는 어렵다. 향후에는 영화·드라마·음식·축제별 catalog를 확대하고, 공식 관광기관·행사 주최기관 URL의 유효 기간과 마지막 검토 시점을 메타데이터로 관리해야 한다.

### 26.2 공식 정보와 원문 근거의 역할 분리

연관 여행지는 “같은 테마로 방문할 수 있는 공식 장소”이지 “해당 작품의 실제 촬영지”가 아니다. 이 둘을 UI와 API에서 계속 구분해야 한다. `source_relation_type`, `direct_location_evidence`, `thematic_destination` 같은 표시를 강화하면 사용자가 연관 추천을 직접 사실 관계로 오해하는 위험을 줄일 수 있다.

### 26.3 최신 실행 수치의 재현성

과거 640개 기반 개선 수치 중 4,189 mentions, 16,674 contexts, semantic OLD/NEW 분포는 이번 문서 작업 범위에서 테스트나 DB를 재실행하지 않았다. 향후 calibration report를 JSON artifact로 버전 관리하거나 실행 ID, 기준 주차, pipeline version, source snapshot을 함께 저장하면 보고서 수치를 자동 재현할 수 있다.

### 26.4 Semantic calibration의 지속 관리

positive/negative anchor와 category matrix는 현재 사례에 맞춰 정교화되어 있다. 새로운 밈, 혼합 언어 작품명, 짧은 지역 브랜드가 들어오면 오탐·미탐 분포가 바뀔 수 있다. gold/negative fixture를 주기적으로 확대하고 카테고리별 precision/recall을 분리 측정해야 한다. Top 50 false positive 0은 중요한 결과지만 모든 주차와 모든 source에서 0을 보장하는 수치는 아니다.

### 26.5 외부 검증 자동화의 경계

현재 Gemini는 검색 query만 제안하고 직접 검색하지 않는다. 실제 촬영지·운영 시간·예약 가능 여부는 외부 검증이 필요하다. 향후 공식 API나 허용된 관광 데이터 provider를 연결할 수 있지만, 검색 결과를 자동으로 사실화하지 않고 provider 신뢰도, URL, 확인 시점, 사람 승인 상태를 Evidence Gate에 포함해야 한다.

### 26.6 Dashboard 캡처 품질

현재 그림 4와 그림 5는 의도한 세부 영역보다 상단 V2 카드가 많이 포함되어 연관 여행지 목록과 Evidence Codes 전체를 한 프레임에서 읽기 어렵다. 최종 발표 자료에는 해당 카드가 실제로 펼쳐진 스크롤 위치에서 별도 캡처를 추가하는 것이 좋다. 다만 이번 보고서는 요청에 따라 기존 최종 PNG를 그대로 참조했고, 이미지를 수정하거나 다시 캡처하지 않았다.

## 27. 결론

Trend Catch V2의 발전은 “Gemini를 붙였다”로 요약할 수 없다. 오히려 반대에 가깝다. V2는 생성형 AI가 판단하기 전에 키워드 품질, 복합 구문 회수, 문장 문맥, 객체 인식, 트렌드-객체 연결, 규칙 필터, 로컬 의미 필터, 정밀도 근거, 고정밀 순위, Evidence Gate를 차례로 적용한다. 최신 Dashboard에서 Raw 2,866개가 Gemini Eligible 1개로 줄어든 결과는 이 설계의 방향을 잘 보여준다.

또한 Related Destination Expansion은 여행 연관성 판별을 실제 서비스 활용 단계로 확장했다. 이제 시스템은 단순히 “F1 더 무비가 여행과 관련 있다”고 말하는 데서 멈추지 않고, 공식 출처가 있는 코리아 인터내셔널 서킷, 인제스피디움, BMW 드라이빙 센터를 테마형 연관 여행지로 제안할 수 있다. 동시에 이 장소들이 실제 촬영지라는 근거는 없으므로 REVIEW와 NEEDS_EVIDENCE를 유지한다. 이 절제가 바로 V2의 핵심 품질이다.

결론적으로 V1이 트렌드를 **발견**했다면 V2는 트렌드를 여행 기회로 **해석하고, 검증하고, 연결**한다. 그리고 그 과정에서 AI의 자유도를 넓히는 대신 입력 후보와 근거를 더 엄격하게 통제함으로써, 여행 콘텐츠 제작에 사용할 수 있는 고정밀·저비용·감사 가능한 Pipeline으로 발전했다.

---

## 부록 A. 주요 구현 근거

| 영역 | 주요 파일·커밋 |
|---|---|
| V2 확정·semantic precision | `app/context_v2/semantic_precision.py`, `8b81d96`, `b580e48` |
| Local E5 | `app/context_v2/embedding_adapter.py`, `semantic_scorer.py`, `400943f` |
| Phrase recall | `app/keywords/phrase_extractor.py`, `phrase_signals.py`, `8ad254a`, `5312d34` |
| KeywordContext/materialization | `app/services/keyword_context_service.py`, `a22305d`, `8963ecc` |
| NER V2 | `app/services/entity_extraction_service.py`, `app/ner/`, `685250b` |
| Ranking/Evidence | `app/services/travel_ranking_service.py`, `0e94626` |
| Gemini final | `app/services/final_travel_opportunity_service.py`, `app/ai/travel_evidence_builder.py`, `e0c4e82` |
| Year in Search | `app/services/google_year_in_search_seed_service.py`, `eed1049` |
| Related Destination | catalog/service/schema/API, `3881725`, `d5cbc41` |
| Dashboard | `dashboard/pages/5_여행_기회_V2.py`, `3e3653b` |

## 부록 B. 검토한 관련 테스트

- `tests/test_keyword_quality_v2.py`
- `tests/test_keyword_phrase_recall_v2.py`
- `tests/test_travel_opportunities_v2.py`
- `tests/test_ner_recall_v2.py`
- `tests/test_travel_semantic_filter_v2_step2.py`
- `tests/test_travel_ranking_v2_step3.py`
- `tests/test_final_travel_opportunity_v2_step4.py`
- `tests/test_related_destination_expansion_v2.py`
- `tests/test_search_interest.py`
- `tests/test_dashboard.py`
- `tests/test_dashboard_client.py`

> 이번 작업은 문서 작성만 수행했으며, 위 테스트를 재실행하지 않았다. 테스트 코드의 검증 의도와 최근 git history만 보고서 근거로 사용했다.
