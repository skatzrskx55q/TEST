import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search, get_model
import torch  # для работы с тензорами

st.set_page_config(page_title="Проверка фраз ФЛ", layout="centered")

# Сначала стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1a6e1a;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .snowflake {
        color: #87CEEB;
        font-size: 1.5rem;
        margin: 0 5px;
        animation: gentleFloat 3s ease-in-out infinite;
        display: inline-block;
    }
    
    @keyframes gentleFloat {
        0%, 100% { 
            transform: translateY(0px) rotate(0deg); 
        }
        50% { 
            transform: translateY(-8px) rotate(180deg); 
        }
    }
    
    .snowflake:nth-child(2n) {
        animation-delay: 0.5s;
    }
    .snowflake:nth-child(3n) {
        animation-delay: 1s;
    }
    .snowflake:nth-child(4n) {
        animation-delay: 1.5s;
    }
    
    .christmas-banner {
        background: linear-gradient(90deg, #1a6e1a, #4caf50, #1a6e1a);
        padding: 12px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 8px rgba(26, 110, 26, 0.3);
    }
    
    .snow-row {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
        margin: 10px 0;
    }

    /* Новогодние карточки для вкладки Да/Нет */
    .christmas-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 20px;
        border-radius: 16px;
        border: 2px solid #4caf50;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.15);
    }
    
    .christmas-card.no {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border: 2px solid #f44336;
        box-shadow: 0 8px 25px rgba(244, 67, 54, 0.15);
    }
    
    .card-snowflake {
        position: absolute;
        color: rgba(255,255,255,0.3);
        font-size: 1rem;
        animation: cardSnowFloat 4s ease-in-out infinite;
    }
    
    @keyframes cardSnowFloat {
        0%, 100% { 
            transform: translateY(0px) rotate(0deg) scale(1); 
            opacity: 0.3; 
        }
        50% { 
            transform: translateY(-10px) rotate(180deg) scale(1.2); 
            opacity: 0.6; 
        }
    }
    
    .card-header {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
        color: #1a6e1a;
    }
    
    .christmas-card.no .card-header {
        color: #c62828;
    }
</style>
""", unsafe_allow_html=True)

# Затем баннер
st.markdown("""
<div class="christmas-banner">
    🎄 С Наступающим Новым Годом! 🎄
</div>
""", unsafe_allow_html=True)

# Затем заголовок со снежинками - разбиваем на части
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Верхний ряд снежинок
    st.markdown("""
    <div class="snow-row">
        <span class="snowflake">❄</span>
        <span class="snowflake">❅</span>
        <span class="snowflake">❆</span>
        <span class="snowflake">•</span>
        <span class="snowflake">❄</span>
        <span class="snowflake">❅</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Нижний ряд иконок
    st.markdown("""
    <div class="snow-row">
        <span class="snowflake">⭐</span>
        <span class="snowflake">🎄</span>
        <span class="snowflake">🎁</span>
        <span class="snowflake">🕯️</span>
        <span class="snowflake">⭐</span>
        <span class="snowflake">🎄</span>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def get_data():
    df = load_all_excels()
    model = get_model()
    df.attrs['phrase_embs'] = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)
    return df

df = get_data()

# 🔘 Все уникальные тематики
all_topics = sorted({topic for topics in df['topics'] for topic in topics})

# --- Вкладки с новогодними иконками ---
tab1, tab2, tab3 = st.tabs(["🎁 Поиск", "🎄 Не используем", "❄️ Да и Нет"])

# ============= TAB 1: ПОИСК =============
with tab1:
    selected_topics = st.multiselect("Фильтр по тематикам (независимо от поиска):", all_topics)
    filter_search_by_topics = st.checkbox("Искать только в выбранных тематиках", value=False)

    # 📂 Фразы по выбранным тематикам
    if selected_topics:
        st.markdown("### 📂 Фразы по выбранным тематикам:")
        filtered_df = df[df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))]
        for row in filtered_df.itertuples():
            with st.container():
                st.markdown(
                    f"""<div style="border: 2px solid #1a6e1a; border-radius: 12px; padding: 16px; margin-bottom: 12px; background: linear-gradient(135deg, #f8fff8 0%, #e8f5e8 100%); box-shadow: 0 2px 6px rgba(26,110,26,0.1);">
                        <div style="font-size: 18px; font-weight: 600; color: #1a472a;">🎁 {row.phrase_full}</div>
                        <div style="margin-top: 4px; font-size: 14px; color: #2e7d32;">🔖 Тематики: <strong>{', '.join(row.topics)}</strong></div>
                    </div>""",
                    unsafe_allow_html=True
                )
                if row.comment and str(row.comment).strip().lower() != "nan":
                    with st.expander("💬 Комментарий", expanded=False):
                        st.markdown(row.comment)

    # 📥 Поисковый запрос
    query = st.text_input("Введите ваш запрос:")

    if query:
        try:
            search_df = df
            if filter_search_by_topics and selected_topics:
                mask = df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))
                search_df = df[mask].copy()

                if not search_df.empty:
                    model = get_model()
                    search_df.attrs['phrase_embs'] = model.encode(search_df['phrase_proc'].tolist(), convert_to_tensor=True)
                else:
                    search_df.attrs['phrase_embs'] = torch.empty((0, 384))

            if search_df.empty:
                st.warning("❄️ Нет данных для поиска по выбранным тематикам.")
            else:
                results = semantic_search(query, search_df)
                if results:
                    st.markdown("### 🎯 Результаты умного поиска:")
                    for score, phrase_full, topics, comment in results:
                        with st.container():
                            if score > 0.8:
                                border_color = "#ffd700"
                                bg_color = "linear-gradient(135deg, #fff9e6 0%, #ffefbf 100%)"
                                icon = "⭐"
                            else:
                                border_color = "#1a6e1a"
                                bg_color = "linear-gradient(135deg, #f8fff8 0%, #e8f5e8 100%)"
                                icon = "🎁"
                            
                            st.markdown(
                                f"""<div style="border: 2px solid {border_color}; border-radius: 12px; padding: 16px; margin-bottom: 12px; background: {bg_color}; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
                                    <div style="font-size: 18px; font-weight: 600; color: #1a472a;">{icon} {phrase_full}</div>
                                    <div style="margin-top: 4px; font-size: 14px; color: #2e7d32;">🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                                    <div style="margin-top: 2px; font-size: 13px; color: #388e3c;">🎯 Релевантность: {score:.2f}</div>
                                </div>""",
                                unsafe_allow_html=True
                            )
                            if comment and str(comment).strip().lower() != "nan":
                                with st.expander("💬 Комментарий", expanded=False):
                                    st.markdown(comment)
                else:
                    st.warning("🎄 Совпадений не найдено в умном поиске.")

                exact_results = keyword_search(query, search_df)
                if exact_results:
                    st.markdown("### 🧷 Точный поиск:")
                    for phrase, topics, comment in exact_results:
                        with st.container():
                            st.markdown(
                                f"""<div style="border: 2px solid #4caf50; border-radius: 12px; padding: 16px; margin-bottom: 12px; background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e8 100%); box-shadow: 0 2px 6px rgba(76,175,80,0.1);">
                                    <div style="font-size: 18px; font-weight: 600; color: #1b5e20;">🎯 {phrase}</div>
                                    <div style="margin-top: 4px; font-size: 14px; color: #2e7d32;">🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                                </div>""",
                                unsafe_allow_html=True
                            )
                            if comment and str(comment).strip().lower() != "nan":
                                with st.expander("💬 Комментарий", expanded=False):
                                    st.markdown(comment)
                else:
                    st.info("❄️ Ничего не найдено в точном поиске.")

        except Exception as e:
            st.error(f"🎄 Ошибка при обработке запроса: {e}")

# ============= TAB 2: НЕ ИСПОЛЬЗУЕМ =============
with tab2:
    st.markdown("""
    <div class="christmas-card no" style="background: linear-gradient(135deg, #fff3e0 0%, #ffebee 100%); border: 2px solid #ff6b6b;">
        <div class="card-snowflake" style="top: 10px; left: 10px; animation-delay: 0s;">❄</div>
        <div class="card-snowflake" style="top: 25px; right: 20px; animation-delay: 1.2s;">❅</div>
        <div class="card-snowflake" style="bottom: 15px; left: 25px; animation-delay: 2.4s;">❆</div>
        <div class="card-header">🎄🚫 Локалы, которые не используем</div>
    """, unsafe_allow_html=True)
    
    unused_topics = [
        "Local_Balance_Transfer", "Local_Friends", "Local_Next_Payment", 
        "Local_Order_Cash", "Local_Other_Cashback", "Local_RemittanceStatus",
        "Подожди (Wait)", "Local_X5", "PassportChangeFirst", "PassportChangeSecond",
        "Меньше (Local_Less)", "Больше (Local_More)", 
        "Рефинансирование под залог недвижимости (Local_Secured_Refinancing)",
        "Действующий займ (Local_Current_MFO_2)", 
        "General Мои кредитные предложения (General_My_loan_offers)",
        "Настроить/Изменить/Восстановить (Local_Setup_Secret_Code)",
        "Как сделать устройство доверенным (Local_Trusted_Device)",
        "Что такое доверенное устройство (Local_About_Trusted_Device)",
        "Что такое секретный код (Local_About_Secret_Code)",
        "займы более 100 тыс (Local_MoreNumbers)", "займы меньше 100 тыс (Local_LessNumbers)",
        "Новая карта (NewCard)", "Проблема с начислением кэшбэка (Local_Problem_CashBack)"
    ]
    
    for topic in unused_topics:
        st.markdown(f"❄️ **{topic}**")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============= TAB 3: ДА/НЕТ =============
def render_phrases_grid(phrases, cols=3, color="#e0f7fa", icon="🎯"):
    rows = [phrases[i:i+cols] for i in range(0, len(phrases), cols)]
    for row in rows:
        cols_streamlit = st.columns(cols)
        for col, phrase in zip(cols_streamlit, row):
            col.markdown(
                f"""<div style="background-color:{color};
                                padding:8px 12px;
                                border-radius:12px;
                                display:inline-block;
                                margin:4px;
                                font-size:14px;
                                border: 1px solid {color};
                                text-align: center;">
                        {icon} {phrase}
                </div>""",
                unsafe_allow_html=True
            )

with tab3:    
    st.markdown("""
    <div class="christmas-card">
        <div class="card-snowflake" style="top: 10px; left: 10px; animation-delay: 0s;">❄</div>
        <div class="card-snowflake" style="top: 15px; right: 15px; animation-delay: 1s;">❅</div>
        <div class="card-snowflake" style="bottom: 20px; left: 20px; animation-delay: 2s;">❆</div>
        <div class="card-header">🎄 Интерпретации 'ДА' 🎄</div>
    """, unsafe_allow_html=True)
    
    yes_phrases = [
        "Подсказать", "Помню", "Хорошо", "Да", "Ага", "Угу",
        "Да по этому вопросу", "Остались", "Можно", "Жги", "Валяй", "Готов",
        "Ну-ну", "Быстрее", "Проверь", "Проверяй", "Все равно хочу",
        "Подскажите", "Расскажи", "Скажи", "Проверил", "Давал",
        "Я могу", "У меня вопрос есть", "Сказал", "Проконсультируйте", "Пробовала вносите в вашу базу"
    ]
    render_phrases_grid(yes_phrases, cols=3, color="#d1f5d3", icon="✅")
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="christmas-card no">
        <div class="card-snowflake" style="top: 10px; left: 15px; animation-delay: 0.5s;">❄</div>
        <div class="card-snowflake" style="top: 25px; right: 10px; animation-delay: 1.5s;">❅</div>
        <div class="card-snowflake" style="bottom: 15px; right: 25px; animation-delay: 2.5s;">❆</div>
        <div class="card-header">🎅 Интерпретации 'НЕТ' 🎅</div>
    """, unsafe_allow_html=True)
    
    no_phrases = [
        "Не надо", "Не хочу", "Не готов", "Не помню", "Не пробовала", "Не интересно"
    ]
    render_phrases_grid(no_phrases, cols=3, color="#f9d6d5", icon="❌")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Новогодний футер
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #1a6e1a; margin-top: 30px;">
        <p>🎄 <strong>С Наступающим Новым Годом!</strong> 🎄</p>
        <div style="font-size: 0.9rem; color: #666;">
            Пусть ваш код всегда будет чистым, а поиск — точным!
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
