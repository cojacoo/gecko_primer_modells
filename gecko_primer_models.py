import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Geoökologie Modelle", layout="wide", page_icon="🌱")

# Custom CSS
st.markdown("""
<style>
.big-font {font-size:20px !important; font-weight: bold;}
.context-box {background-color: #e8f4f8; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4;}
.insight-box {background-color: #fff4e6; padding: 15px; border-radius: 10px; border-left: 5px solid #ff7f0e;}
.warning-box {background-color: #ffe6e6; padding: 15px; border-radius: 10px; border-left: 5px solid #d62728;}
</style>
""", unsafe_allow_html=True)

# Titel
st.title("🌱 Jeder ist ein Modellierer: Von der Beobachtung zum Verständnis")
st.markdown("**Geoökologische Grundlagen - TU Bergakademie Freiberg**")

# Sidebar Navigation
st.sidebar.header("🧭 Deine Modellierungsreise")
schritt = st.sidebar.radio(
    "Wähle deinen Schritt:",
    ["🔍 Schritt 1: Muster finden (Regression)",
     "📦 Schritt 2: Akkumulation verstehen (Stock-Flow → Logistik)",
     "⏱️ Schritt 3: Schnelle & langsame Dynamik (simpleLSTM)",
     "🔄 Schritt 4: Wenn Rückkopplungen übernehmen (Vegetation-Boden)",
     "🦊 Schritt 5: Interaktionen & Koexistenz (Räuber-Beute)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**💡 Die Reise:**
1. **Beobachten** - Zusammenhänge
2. **Verstehen** - Prozesse
3. **Erkennen** - Zeitskalen
4. **Vorhersagen** - Rückkopplungen
5. **Analysieren** - Interaktionen
""")

# ============================================================================
# SCHRITT 1: REGRESSION
# ============================================================================
if "Schritt 1" in schritt:
    st.header("🔍 Schritt 1: Muster in Daten finden")
    
    st.markdown("""
    <div class="context-box">
    <h3>🌲 Die Förster-Frage</h3>
    <p><strong>Du bist Förster.</strong> Deine Frage: <em>Was kontrolliert, wie viel mein Wald jährlich wächst?</em></p>
    <p><strong>NPP (Netto-Primärproduktion)</strong> = Kohlenstoff, den der Wald pro Jahr in neuer Biomasse speichert.</p>
    <p><strong>Deine Hypothese:</strong> Temperatur, Niederschlag und Boden-Stickstoff sind wichtig. Aber wie stark? Und wirken sie additiv?</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Fixe Daten generieren
    if 'daten_generiert' not in st.session_state:
        np.random.seed(42)
        n_punkte = 60
        
        T_daten = np.random.uniform(6, 14, n_punkte)
        P_daten = np.random.uniform(600, 1400, n_punkte)
        N_daten = np.random.uniform(8, 18, n_punkte)
        
        NPP_wahr = -100 + 35*T_daten + 0.7*P_daten + 12*N_daten
        NPP_beob = NPP_wahr + np.random.normal(0, 45, n_punkte)
        
        st.session_state.daten_generiert = True
        st.session_state.T_daten = T_daten
        st.session_state.P_daten = P_daten
        st.session_state.N_daten = N_daten
        st.session_state.NPP_beob = NPP_beob
    
    T_daten = st.session_state.T_daten
    P_daten = st.session_state.P_daten
    N_daten = st.session_state.N_daten
    NPP_beob = st.session_state.NPP_beob
    
    tab1, tab2, tab3 = st.tabs(["📊 Einzeleffekte", "➕ Kombiniertes Modell", "🎯 Dein Modell"])
    
    with tab1:
        st.subheader("Jede Variable separat testen")
        st.markdown("**💡 Währung: STOFF (C) + ENERGIE (T) + WASSER (P)**")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Get overall y limits from data
        y_min = NPP_beob.min() - 50
        y_max = NPP_beob.max() + 50
        
        # Temperatur
        z = np.polyfit(T_daten, NPP_beob, 1)
        p = np.poly1d(z)
        T_linie = np.linspace(T_daten.min(), T_daten.max(), 100)
        NPP_pred_T = p(T_daten)
        r2_T = r2_score(NPP_beob, NPP_pred_T)
        
        # Calculate confidence intervals (95%)
        residuals = NPP_beob - NPP_pred_T
        std_err = np.std(residuals)
        ci_T = 1.96 * std_err
        
        axes[0].scatter(T_daten, NPP_beob, alpha=0.6, s=60, edgecolor='black', linewidth=0.5)
        axes[0].plot(T_linie, p(T_linie), 'r-', lw=2.5, label=f'R² = {r2_T:.2f}')
        axes[0].fill_between(T_linie, p(T_linie)-ci_T, p(T_linie)+ci_T, alpha=0.2, color='red', label='95% KI')
        axes[0].set_xlabel('Temperatur T (°C)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('NPP (g C/m²/Jahr)', fontsize=11, fontweight='bold')
        axes[0].set_title('Temperatureffekt (ENERGIE-Währung)', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_ylim(y_min, y_max)
        
        # Niederschlag
        z = np.polyfit(P_daten, NPP_beob, 1)
        p = np.poly1d(z)
        P_linie = np.linspace(P_daten.min(), P_daten.max(), 100)
        NPP_pred_P = p(P_daten)
        r2_P = r2_score(NPP_beob, NPP_pred_P)
        
        residuals = NPP_beob - NPP_pred_P
        std_err = np.std(residuals)
        ci_P = 1.96 * std_err
        
        axes[1].scatter(P_daten, NPP_beob, alpha=0.6, s=60, edgecolor='black', linewidth=0.5)
        axes[1].plot(P_linie, p(P_linie), 'r-', lw=2.5, label=f'R² = {r2_P:.2f}')
        axes[1].fill_between(P_linie, p(P_linie)-ci_P, p(P_linie)+ci_P, alpha=0.2, color='red', label='95% KI')
        axes[1].set_xlabel('Niederschlag P (mm/Jahr)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('NPP (g C/m²/Jahr)', fontsize=11, fontweight='bold')
        axes[1].set_title('Niederschlagseffekt (WASSER-Währung)', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim(y_min, y_max)
        
        # Stickstoff
        z = np.polyfit(N_daten, NPP_beob, 1)
        p = np.poly1d(z)
        N_linie = np.linspace(N_daten.min(), N_daten.max(), 100)
        NPP_pred_N = p(N_daten)
        r2_N = r2_score(NPP_beob, NPP_pred_N)
        
        residuals = NPP_beob - NPP_pred_N
        std_err = np.std(residuals)
        ci_N = 1.96 * std_err
        
        axes[2].scatter(N_daten, NPP_beob, alpha=0.6, s=60, edgecolor='black', linewidth=0.5)
        axes[2].plot(N_linie, p(N_linie), 'r-', lw=2.5, label=f'R² = {r2_N:.2f}')
        axes[2].fill_between(N_linie, p(N_linie)-ci_N, p(N_linie)+ci_N, alpha=0.2, color='red', label='95% KI')
        axes[2].set_xlabel('Boden-Stickstoff N (g/m²)', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('NPP (g C/m²/Jahr)', fontsize=11, fontweight='bold')
        axes[2].set_title('Stickstoffeffekt (STOFF-Währung)', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        axes[2].set_ylim(y_min, y_max)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>🔍 Was sehen wir?</strong>
        <ul>
        <li>Alle drei Variablen zeigen positive Zusammenhänge mit NPP</li>
        <li>Aber R²-Werte sind niedrig ({r2_T:.2f}, {r2_P:.2f}, {r2_N:.2f}) - viel Streuung!</li>
        <li>Warum? Weil sie <strong>zusammen wirken</strong>, nicht allein</li>
        </ul>
        <strong>Nächster Schritt:</strong> Kombiniere sie in einem Modell!
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Multiple Regression: Das vollständige Modell")
        st.markdown("**NPP = β₀ + β₁×T + β₂×P + β₃×N + ε**")
        
        X = np.column_stack([T_daten, P_daten, N_daten])
        modell = LinearRegression()
        modell.fit(X, NPP_beob)
        
        beta0 = modell.intercept_
        beta1, beta2, beta3 = modell.coef_
        NPP_vorher = modell.predict(X)
        r2_voll = r2_score(NPP_beob, NPP_vorher)
        rmse_voll = np.sqrt(mean_squared_error(NPP_beob, NPP_vorher))
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**📊 Modellkoeffizienten:**")
            st.metric("β₀ (Achsenabschnitt)", f"{beta0:.0f}")
            st.metric("β₁ (Temperatur)", f"{beta1:.1f} g C/m²/°C")
            st.metric("β₂ (Niederschlag)", f"{beta2:.2f} g C/m²/mm")
            st.metric("β₃ (Stickstoff)", f"{beta3:.1f} g C/(m²·gN/m²)")
            
            st.markdown("**✅ Modellgüte:**")
            st.metric("R²", f"{r2_voll:.3f}")
            st.metric("RMSE", f"{rmse_voll:.1f} g C/m²/Jahr")
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(NPP_vorher, NPP_beob, alpha=0.6, s=60, edgecolor='black', linewidth=0.5)
            lim_min = min(NPP_vorher.min(), NPP_beob.min())
            lim_max = max(NPP_vorher.max(), NPP_beob.max())
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2, label='Perfekte Anpassung')
            ax.set_xlabel('Vorhergesagte NPP (g C/m²/Jahr)', fontsize=12)
            ax.set_ylabel('Beobachtete NPP (g C/m²/Jahr)', fontsize=12)
            ax.set_title(f'Modellgüte (R² = {r2_voll:.3f})', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
            ax.set_aspect('equal')
            st.pyplot(fig)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>🎯 Wichtige Erkenntnisse:</strong>
        <ul>
        <li><strong>R² = {r2_voll:.3f}</strong> - Viel besser als Einzelmodelle! (war ~{r2_T:.2f})</li>
        <li><strong>Additive Effekte:</strong> Jede Variable trägt unabhängig bei</li>
        <li><strong>Temperatur wichtigster Faktor:</strong> β₁ = {beta1:.1f} (1°C wärmer → +{beta1:.0f} g C/m²/Jahr)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("🎮 Baue dein eigenes Modell")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**🎚️ Deine Modellparameter:**")
            user_beta0 = st.slider("β₀ (Achsenabschnitt)", -300, 100, int(beta0), 25)
            user_beta1 = st.slider("β₁ (Temperatur)", 0, 80, int(beta1), 5)
            user_beta2 = st.slider("β₂ (Niederschlag)", 0.0, 2.0, float(beta2), 0.1)
            user_beta3 = st.slider("β₃ (Stickstoff)", 0, 30, int(beta3), 2)
            
            NPP_user = user_beta0 + user_beta1*T_daten + user_beta2*P_daten + user_beta3*N_daten
            r2_user = r2_score(NPP_beob, NPP_user)
            rmse_user = np.sqrt(mean_squared_error(NPP_beob, NPP_user))
            
            st.markdown("**📈 Deine Modellgüte:**")
            r2_color = "normal" if r2_user < r2_voll - 0.1 else "inverse" if r2_user > r2_voll - 0.02 else "off"
            st.metric("R²", f"{r2_user:.3f}", f"{(r2_user-r2_voll):.3f}", delta_color=r2_color)
            st.metric("RMSE", f"{rmse_user:.1f}", f"{(rmse_user-rmse_voll):.1f}")
            
            if r2_user > r2_voll - 0.02:
                st.success("🎉 Exzellent! Du bist sehr nah am optimalen Modell!")
            elif r2_user > r2_voll - 0.1:
                st.info("👍 Gut! Versuche zu verfeinern um R² zu verbessern")
            else:
                st.warning("🤔 Dein Modell kann verbessert werden!")
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(NPP_user, NPP_beob, alpha=0.6, s=60, edgecolor='black', linewidth=0.5,
                      c=np.abs(NPP_beob - NPP_user), cmap='RdYlGn_r', vmin=0, vmax=200)
            
            lim_min, lim_max = 300, 1100
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2, label='Perfekte Anpassung')
            ax.scatter(NPP_vorher, NPP_beob, alpha=0.3, s=40, edgecolor='blue',
                      facecolor='none', linewidth=1.5, label='Optimales Modell')
            
            ax.set_xlabel('Vorhergesagte NPP (g C/m²/Jahr)', fontsize=12)
            ax.set_ylabel('Beobachtete NPP (g C/m²/Jahr)', fontsize=12)
            ax.set_title(f'Dein Modell: R² = {r2_user:.3f}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)
            ax.set_aspect('equal')
            
            cbar = plt.colorbar(ax.collections[0], ax=ax)
            cbar.set_label('Vorhersagefehler', fontsize=10)
            
            st.pyplot(fig)
        
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Aber Moment...</strong> Dieses Modell hat ein Problem!
        <p>Was wenn die Temperatur weiter steigt? NPP steigt unbegrenzt?</p>
        <p><strong>In Wirklichkeit:</strong> Wachstum ist begrenzt! Es gibt eine Tragfähigkeit.</p>
        <p><strong>Nächster Schritt:</strong> Wir müssen <em>Akkumulation mit Grenzen</em> verstehen.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# SCHRITT 2: STOCK-FLOW → LOGISTIK
# ============================================================================
elif "Schritt 2" in schritt:
    st.header("📦 Schritt 2: Wachstum mit Grenzen verstehen")
    
    st.markdown("""
    <div class="context-box">
    <h3>🌳 Von jährlichem Wachstum zu Gesamtbiomasse</h3>
    <p><strong>Die nächste Förster-Frage:</strong> <em>Wie viel Gesamtbiomasse erreicht mein Wald?</em></p>
    <p><strong>Das Problem:</strong> NPP (Wachstum pro Jahr) ist eine Sache, aber Gesamtbiomasse akkumuliert über Zeit.</p>
    <p><strong>Die Erkenntnis:</strong> Wenn der Wald wächst, verlangsamt sich das Wachstum (mehr Konkurrenz, Beschattung, etc.)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📥 Einfache Akkumulation", "🛑 Grenze hinzufügen", "📊 Das logistische Modell"])
    
    with tab1:
        st.subheader("Versuch 1: Einfache Akkumulation (Stock-Flow)")
        st.markdown("**Modell:** dC/dt = NPP (konstant)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            NPP_konst = st.slider("NPP (g C/m²/Jahr)", 200, 1000, 500, 50)
            C0_einfach = st.slider("Anfangsbiomasse C₀ (kg C/m²)", 0.0, 5.0, 1.0, 0.5)
            t_max_einfach = st.slider("Zeit (Jahre)", 20, 100, 50, 10)
            
            C_ende_einfach = C0_einfach + (NPP_konst * t_max_einfach / 1000)
            st.metric(f"Biomasse nach {t_max_einfach} Jahren", f"{C_ende_einfach:.1f} kg C/m²")
        
        with col2:
            t = np.linspace(0, t_max_einfach, 500)
            C_einfach = C0_einfach + (NPP_konst / 1000) * t
            
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(t, C_einfach, 'b-', lw=3, label='Einfache Akkumulation')
            ax.set_xlabel('Zeit (Jahre)', fontsize=12)
            ax.set_ylabel('Gesamtbiomasse (kg C/m²)', fontsize=12)
            ax.set_title('Einfaches Stock-Flow Modell', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, max(C_einfach)*1.2)
            st.pyplot(fig)
        
        st.markdown(f"""
        <div class="warning-box">
        <strong>❌ Das Problem:</strong>
        <ul>
        <li>Biomasse wächst linear für immer!</li>
        <li>In 100 Jahren: {C0_einfach + (NPP_konst * 100 / 1000):.0f} kg C/m² - das ist absurd!</li>
        <li>Echte Wälder erreichen eine maximale Höhe/Dichte</li>
        </ul>
        <strong>Wir brauchen eine Grenze...</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Versuch 2: Wachstum verlangsamt sich bei Annäherung an Grenze")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            r_exp = st.slider("Wachstumsrate r (1/Jahr)", 0.05, 0.5, 0.15, 0.05)
            K_tragf = st.slider("Tragfähigkeit K (kg C/m²)", 5.0, 30.0, 15.0, 2.5)
            C0_exp = st.slider("Anfangsbiomasse C₀", 0.5, 5.0, 1.0, 0.5)
            t_max_exp = st.slider("Simulationszeit (Jahre)", 20, 150, 80, 10)
            
            t_90 = np.log(9) / r_exp
            
            st.metric("Tragfähigkeit K", f"{K_tragf:.1f} kg C/m²")
            st.metric("Zeit bis 90% von K", f"{t_90:.1f} Jahre")
        
        with col2:
            def logistisches_wachstum(C, t):
                return r_exp * C * (1 - C/K_tragf)
            
            t = np.linspace(0, t_max_exp, 500)
            C_log = odeint(logistisches_wachstum, C0_exp, t).flatten()
            
            wachstumsrate = r_exp * C_log * (1 - C_log/K_tragf)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))
            
            ax1.plot(t, C_log, 'g-', lw=3, label='Logistisches Wachstum')
            ax1.axhline(K_tragf, ls='--', color='red', lw=2, label=f'Tragfähigkeit K={K_tragf:.1f}')
            ax1.axhline(K_tragf*0.9, ls=':', color='gray', alpha=0.7)
            ax1.axvline(t_90, ls=':', color='gray', alpha=0.7, label=f'90% bei t={t_90:.0f} J')
            ax1.fill_between(t, 0, K_tragf, alpha=0.1, color='green')
            ax1.set_xlabel('Zeit (Jahre)', fontsize=12)
            ax1.set_ylabel('Gesamtbiomasse (kg C/m²)', fontsize=12)
            ax1.set_title('Logistisches Wachstum: Biomasse-Akkumulation', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(alpha=0.3)
            
            ax2.plot(t, wachstumsrate, 'purple', lw=3)
            ax2.axhline(0, color='black', lw=1)
            ax2.set_xlabel('Zeit (Jahre)', fontsize=12)
            ax2.set_ylabel('Wachstumsrate dC/dt (kg C/m²/Jahr)', fontsize=12)
            ax2.set_title('Wie sich die Wachstumsrate ändert', fontsize=14, fontweight='bold')
            ax2.grid(alpha=0.3)
            
            max_wachstum_idx = np.argmax(wachstumsrate)
            ax2.scatter(t[max_wachstum_idx], wachstumsrate[max_wachstum_idx],
                       s=200, c='red', zorder=10, edgecolor='black', linewidth=2)
            ax2.annotate(f'Max Wachstum bei C=K/2\n({C_log[max_wachstum_idx]:.1f} kg C/m²)',
                        xy=(t[max_wachstum_idx], wachstumsrate[max_wachstum_idx]),
                        xytext=(t[max_wachstum_idx]+10, wachstumsrate[max_wachstum_idx]*0.8),
                        fontsize=10, ha='left',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', lw=1.5))
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("""
        <div class="insight-box">
        <strong>✅ Jetzt macht es Sinn!</strong>
        <ul>
        <li><strong>Früh:</strong> Schnelles exponentielles Wachstum (viel Platz/Licht)</li>
        <li><strong>Mitte:</strong> Maximale Wachstumsrate bei C = K/2</li>
        <li><strong>Spät:</strong> Wachstum verlangsamt sich bei Annäherung an K (Konkurrenz, Beschattung)</li>
        <li><strong>Gleichgewicht:</strong> Bei K, Wachstum = 0 (ausgeglichen durch Mortalität)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Das vollständige logistische Modell")
        st.markdown("**dC/dt = r×C×(1 - C/K)**")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            r_final = st.slider("Wachstumsrate r", 0.05, 0.8, 0.2, 0.05)
            K_final = st.slider("Tragfähigkeit K", 8.0, 25.0, 15.0, 1.0)
            C0_niedrig = st.slider("Start 1 (unter K)", 0.5, 8.0, 2.0, 0.5)
            C0_hoch = st.slider("Start 2 (über K)", 10.0, 30.0, 20.0, 2.0)
            
            st.markdown(f"""
            <div class="context-box">
            <strong>🎯 Die Geschichte:</strong>
            <p>Zwei Wälder:</p>
            <ul>
            <li><strong>Jungwald</strong> (C₀ = {C0_niedrig:.1f}): Wächst auf</li>
            <li><strong>Überalterter Wald</strong> (C₀ = {C0_hoch:.1f}): Nimmt ab</li>
            </ul>
            <p>Beide konvergieren zu K = {K_final:.1f} kg C/m²</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            def logistik_final(C, t):
                return r_final * C * (1 - C/K_final)
            
            t = np.linspace(0, 60, 500)
            C1 = odeint(logistik_final, C0_niedrig, t).flatten()
            C2 = odeint(logistik_final, C0_hoch, t).flatten()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            ax1.plot(t, C1, 'b-', lw=3, label=f'Jungwald (C₀={C0_niedrig:.1f})')
            ax1.plot(t, C2, 'r-', lw=3, label=f'Überaltert (C₀={C0_hoch:.1f})')
            ax1.axhline(K_final, ls='--', color='green', lw=2, label=f'Gleichgewicht K={K_final:.1f}')
            ax1.fill_between(t, K_final*0.95, K_final*1.05, alpha=0.15, color='green')
            ax1.set_xlabel('Zeit (Jahre)', fontsize=12)
            ax1.set_ylabel('Biomasse C (kg C/m²)', fontsize=12)
            ax1.set_title('Beide konvergieren zum Gleichgewicht', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(alpha=0.3)
            
            C_bereich = np.linspace(0, K_final*1.5, 300)
            dC_dt = r_final * C_bereich * (1 - C_bereich/K_final)
            
            ax2.plot(C_bereich, dC_dt, 'purple', lw=3)
            ax2.axhline(0, color='black', lw=1)
            ax2.axvline(K_final, ls='--', color='green', lw=2, alpha=0.7)
            ax2.fill_between(C_bereich, 0, dC_dt, where=(dC_dt>0), alpha=0.2, color='blue', label='Wachstumszone')
            ax2.fill_between(C_bereich, 0, dC_dt, where=(dC_dt<0), alpha=0.2, color='red', label='Rückgangszone')
            
            ax2.scatter([0, K_final], [0, 0], s=[150, 300], c=['red', 'green'],
                       edgecolors='black', linewidths=2, zorder=10)
            ax2.text(K_final, -0.05*dC_dt.max(), 'Stabiles\nGleichgewicht', ha='center', va='top',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            ax2.set_xlabel('Biomasse C (kg C/m²)', fontsize=12)
            ax2.set_ylabel('Wachstumsrate dC/dt', fontsize=12)
            ax2.set_title('Phasendiagramm: Negative Rückkopplung', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10, loc='upper right')
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("""
        <div class="insight-box">
        <strong>🧠 Konzeptionelle Haupterkenntnis:</strong>
        <p><strong>Stock-Flow Denken führte uns zum logistischen Modell!</strong></p>
        <ul>
        <li><strong>Bestand (C):</strong> Akkumulierte Gesamtbiomasse</li>
        <li><strong>Zufluss:</strong> Wachstum = r×C (proportional zur aktuellen Biomasse)</li>
        <li><strong>Abfluss (implizit):</strong> Begrenzt durch Tragfähigkeit</li>
        <li><strong>Nettofluss:</strong> dC/dt = r×C×(1 - C/K)</li>
        </ul>
        <p><strong>Dies ist NEGATIVE RÜCKKOPPLUNG:</strong> Höheres C → Niedrigeres dC/dt</p>
        <p><strong>Ergebnis:</strong> Selbststabilisierung bei K</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>⏱️ Aber es gibt mehr zur Geschichte...</strong>
        <p>Echte Ökosysteme haben <strong>mehrere Zeitskalen</strong>:</p>
        <ul>
        <li>Schnell: Blätter wachsen und fallen (Monate)</li>
        <li>Langsam: Boden baut sich auf (Jahrzehnte bis Jahrhunderte)</li>
        </ul>
        <p><strong>Weiter:</strong> Schnelle & langsame Dynamik verstehen (simpleLSTM)</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# SCHRITT 3: simpleLSTM
# ============================================================================
elif "Schritt 3" in schritt:
    st.header("⏱️ Schritt 3: Schnelle & Langsame Dynamik (simpleLSTM)")
    
    st.markdown("""
    <div class="context-box">
    <h3>🌾 Das Landwirt-Dilemma</h3>
    <p><strong>Du bist Landwirt mit konventioneller Bodenbearbeitung.</strong></p>
    <p><strong>Deine Beobachtung:</strong> Jedes Jahr fügst du Pflanzenreste hinzu (Stroh, Wurzeln). Sie verschwinden schnell.
    Aber dein Bodenkohlenstoff? Der ändert sich kaum.</p>
    <p><strong>Das Rätsel:</strong> Du fügst 5 Tonnen/ha jedes Jahr hinzu. Wo geht es hin?</p>
    <p><strong>Die Erkenntnis:</strong> Es gibt ZWEI Kohlenstoffpools mit sehr unterschiedlichen Zeitskalen!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🔬 Die zwei Pools", "⚡ Landwirtschaftliche Szenarien", "🎯 Resilienz & Kollaps"])
    
    with tab1:
        st.subheader("Schnellen vs. Langsamen Kohlenstoff verstehen")
        st.markdown("**💡 Währung: STOFF (C) | Werkzeug: Flüsse & Speicher + Zeitskalen**")
        
        st.markdown("""
        **Die zwei Zeitskalen:**
        - **SCHNELL (Tage bis Jahre):** Ernterückstände, frische organische Substanz → hoher Umsatz
        - **LANGSAM (Jahrzehnte bis Jahrhunderte):** Humus, stabile organische Substanz → niedriger Umsatz
        
        **Die Gleichungen:**
        ```
        dC_schnell/dt = f × Input - k_schnell × C_schnell
                        ↑ Zufuhr      ↑ Schneller Abbau
        
        dC_langsam/dt = (1-f) × Input + ε × k_schnell × C_schnell - k_langsam × C_langsam
                        ↑ Direkt       ↑ Transfer             ↑ Langsamer Abbau
        ```
        
        **Variablen:**
        - `C_schnell` = Schneller Pool [Mg C/ha] (z.B. Stroh, Wurzeln)
        - `C_langsam` = Langsamer Pool [Mg C/ha] (z.B. Humus)
        - `f` = Fraktion zu schnellem Pool [0-1]
        - `k_schnell` = Schnelle Abbaurate [1/Jahr] (groß!)
        - `k_langsam` = Langsame Abbaurate [1/Jahr] (klein!)
        - `ε` = Transfer-Effizienz [0-1] (wieviel vom schnellen Pool wird zu langsamem)
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**🌾 Pflanzenrest-Input:**")
            rest_input = st.slider("Jährliche Reste (Mg C/ha/Jahr)", 1.0, 6.0, 3.5, 0.5)
            
            st.markdown("**⚡ Schneller Pool (Reste, τ ~ 1 Jahr):**")
            fraktion_schnell = st.slider("Fraktion zu schnellem Pool", 0.5, 0.9, 0.8, 0.05)
            k_schnell = st.slider("Schnelle Abbaurate k_s (1/Jahr)", 0.5, 2.5, 1.2, 0.1)
            tau_schnell = 1/k_schnell
            
            st.markdown("**🐌 Langsamer Pool (Humus, τ ~ 50 Jahre):**")
            k_langsam = st.slider("Langsame Abbaurate k_l (1/Jahr)", 0.01, 0.05, 0.02, 0.005)
            tau_langsam = 1/k_langsam
            transfer_eff = st.slider("Schnell→Langsam Effizienz ε", 0.1, 0.4, 0.25, 0.05)
            
            st.markdown("---")
            st.markdown("**⏱️ Zeitskalentrennung (simpleLSTM!):**")
            st.metric("τ_schnell (Verweilzeit)", f"{tau_schnell:.1f} Jahre", help="1/k_schnell")
            st.metric("τ_langsam (Verweilzeit)", f"{tau_langsam:.0f} Jahre", help="1/k_langsam")
            trennung = tau_langsam/tau_schnell
            st.metric("Verhältnis τ_l/τ_s", f"{trennung:.0f}×", 
                     delta="Stark" if trennung > 20 else "Schwach",
                     delta_color="normal" if trennung > 20 else "inverse")
            
            if trennung > 20:
                st.success("✅ Starke simpleLSTM Trennung!")
            else:
                st.warning("⚠️ Schwache Trennung - vergrößere das Verhältnis!")
        
        with col2:
            def lstm_kohlenstoff(y, t):
                C_schnell, C_langsam = y
                dC_schnell = fraktion_schnell * rest_input - k_schnell * C_schnell
                dC_langsam = (1 - fraktion_schnell) * rest_input + transfer_eff * k_schnell * C_schnell - k_langsam * C_langsam
                return [dC_schnell, dC_langsam]
            
            t = np.linspace(0, 150, 1500)
            y0 = [0, 0]
            sol = odeint(lstm_kohlenstoff, y0, t)
            
            C_schnell_eq = fraktion_schnell * rest_input / k_schnell
            C_langsam_eq = ((1-fraktion_schnell)*rest_input + transfer_eff*k_schnell*C_schnell_eq) / k_langsam
            C_total_eq = C_schnell_eq + C_langsam_eq
            
            # FIXE ACHSEN für Vergleichbarkeit
            y_max_fixed = max(100, C_total_eq * 1.15)
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 13))
            
            # Panel 1: Gesamtübersicht
            ax1.plot(t, sol[:, 0], 'orange', lw=3, label=f'⚡ Schneller Pool (τ={tau_schnell:.1f} J)')
            ax1.plot(t, sol[:, 1], 'brown', lw=3.5, label=f'🐌 Langsamer Pool (τ={tau_langsam:.0f} J)')
            ax1.plot(t, sol[:, 0] + sol[:, 1], 'black', lw=3, label='📊 Gesamt-SOC', linestyle='--')
            ax1.axhline(C_schnell_eq, ls=':', color='orange', alpha=0.5)
            ax1.axhline(C_langsam_eq, ls=':', color='brown', alpha=0.5)
            ax1.axhline(C_total_eq, ls=':', color='black', alpha=0.5)
            
            # Annotate equilibria
            ax1.text(t[-1]*0.98, C_schnell_eq, f'  {C_schnell_eq:.1f}', va='center', ha='right', 
                    fontsize=10, color='orange', fontweight='bold')
            ax1.text(t[-1]*0.98, C_langsam_eq, f'  {C_langsam_eq:.1f}', va='center', ha='right',
                    fontsize=10, color='brown', fontweight='bold')
            
            ax1.set_xlabel('Zeit (Jahre)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Bodenkohlenstoff (Mg C/ha)', fontsize=12, fontweight='bold')
            ax1.set_title('simpleLSTM: Zwei Zeitskalen im Gesamtsystem', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11, loc='right')
            ax1.grid(alpha=0.3)
            ax1.set_ylim(0, y_max_fixed)
            ax1.set_xlim(0, 150)
            
            # Panel 2: Zoom erste 20 Jahre (FIXE ACHSEN)
            t_zoom = 20
            idx_zoom = t <= t_zoom
            ax2.plot(t[idx_zoom], sol[idx_zoom, 0], 'orange', lw=4, 
                    label=f'⚡ Schnell: Äquilibriert in ~{3*tau_schnell:.0f} Jahren')
            ax2.plot(t[idx_zoom], sol[idx_zoom, 1], 'brown', lw=4,
                    label=f'🐌 Langsam: Akkumuliert weiter (braucht ~{3*tau_langsam:.0f} Jahre)')
            ax2.axhline(C_schnell_eq, ls='--', color='orange', lw=2, alpha=0.7, 
                       label=f'Schnelles Gleichgewicht')
            
            # Mark where fast pool reaches 95% of equilibrium
            t_95_schnell = -np.log(0.05) / k_schnell
            if t_95_schnell < t_zoom:
                ax2.axvline(t_95_schnell, ls=':', color='orange', lw=2, alpha=0.5)
                ax2.text(t_95_schnell, y_max_fixed*0.9, f'95% erreicht\n({t_95_schnell:.1f} J)', 
                        ha='center', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax2.set_xlabel('Zeit (Jahre)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Bodenkohlenstoff (Mg C/ha)', fontsize=12, fontweight='bold')
            ax2.set_title(f'ZOOM: Erste {t_zoom} Jahre - Schneller Pool äquilibriert!', 
                         fontsize=14, fontweight='bold', color='darkorange')
            ax2.legend(fontsize=10, loc='lower right')
            ax2.grid(alpha=0.3)
            ax2.set_ylim(0, y_max_fixed)  # GLEICHE ACHSE!
            ax2.set_xlim(0, t_zoom)
            
            # Panel 3: Nur schneller Pool für Klarheit
            ax3.plot(t[idx_zoom], sol[idx_zoom, 0], 'orange', lw=5, label='Schneller Pool')
            ax3.axhline(C_schnell_eq, ls='--', color='red', lw=2, label=f'Gleichgewicht = {C_schnell_eq:.2f} Mg C/ha')
            ax3.fill_between(t[idx_zoom], 0, sol[idx_zoom, 0], alpha=0.3, color='orange')
            
            # Show exponential approach
            ax3.text(t_zoom*0.5, C_schnell_eq*0.5, 
                    f'Exponentieller\nAnsatz:\nτ = {tau_schnell:.1f} Jahre',
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='orange', linewidth=2))
            
            ax3.set_xlabel('Zeit (Jahre)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Schneller Pool (Mg C/ha)', fontsize=12, fontweight='bold')
            ax3.set_title('FOKUS: Schneller Pool - Sichtbarer Effekt!', fontsize=14, fontweight='bold', color='darkorange')
            ax3.legend(fontsize=11)
            ax3.grid(alpha=0.3)
            ax3.set_ylim(0, max(C_schnell_eq * 1.2, 5))  # Eigene Skala für Sichtbarkeit
            ax3.set_xlim(0, t_zoom)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>🧠 simpleLSTM Kernaussagen:</strong>
        <ul>
        <li><strong>Zwei Pools, zwei Zeitskalen:</strong>
            <ul>
            <li>⚡ Schneller Pool: {C_schnell_eq:.2f} Mg C/ha, τ = {tau_schnell:.1f} Jahre → <strong>Äquilibriert schnell!</strong></li>
            <li>🐌 Langsamer Pool: {C_langsam_eq:.1f} Mg C/ha, τ = {tau_langsam:.0f} Jahre → <strong>Langzeitgedächtnis!</strong></li>
            </ul>
        </li>
        <li><strong>Systemgedächtnis:</strong> Der langsame Pool "erinnert" sich über Jahrzehnte - daher reagieren Böden träge!</li>
        <li><strong>Management-Implikation:</strong> 
            <ul>
            <li>Schneller Pool: Reagiert sofort auf Änderungen (z.B. Mulchen)</li>
            <li>Langsamer Pool: Braucht Jahrzehnte zum Auf-/Abbau (z.B. Humus)</li>
            <li>Gesamt-SOC wird vom langsamen Pool dominiert ({C_langsam_eq/C_total_eq*100:.0f}% der Gesamt-SOC!)</li>
            </ul>
        </li>
        <li><strong>Zeitskalentrennung:</strong> Verhältnis {trennung:.0f}× bedeutet: Der schnelle Pool ist auf Jahrhundertskala "unsichtbar"</li>
        </ul>
        <p><strong>🔗 Werkzeug-Verbindung:</strong> Dies ist <em>Werkzeug #2: Skalen & Hierarchien</em> - 
        verschiedene Prozesse auf verschiedenen Zeitskalen!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>🧠 simpleLSTM Erkenntnis:</strong>
        <ul>
        <li><strong>Schneller Pool:</strong> Pflanzenreste zersetzen sich in ~{tau_schnell:.1f} Jahren</li>
        <li><strong>Langsamer Pool:</strong> Humus persistiert ~{tau_langsam:.0f} Jahre</li>
        <li><strong>Im Gleichgewicht:</strong>
            <ul>
            <li>Schneller Pool: {C_schnell_eq:.2f} Mg C/ha (turniert schnell)</li>
            <li>Langsamer Pool: {C_langsam_eq:.1f} Mg C/ha (meiste SOC!)</li>
            <li>Gesamt: {C_schnell_eq + C_langsam_eq:.1f} Mg C/ha</li>
            </ul>
        </li>
        <li><strong>Zeitskalentrennung:</strong> {tau_langsam/tau_schnell:.0f}× bedeutet schneller Pool "unsichtbar" auf langen Zeitskalen</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("🚜 Landwirtschaftliche Management-Szenarien")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Szenario 1: Konventionelle Bodenbearbeitung**")
            input_konv = st.slider("Rest-Input (Mg C/ha/J)", 1.0, 4.0, 2.5, 0.5, key='input_konv')
            k_langsam_konv = st.slider("Langsamer Abbau (Bodenbearbeitungseffekt)", 0.04, 0.12, 0.08, 0.01, key='k_langsam_konv')
            
            st.markdown("**Szenario 2: Direktsaat mit Zwischenfrüchten**")
            input_direkt = st.slider("Rest-Input (höher!)", 2.0, 6.0, 4.0, 0.5, key='input_direkt')
            k_langsam_direkt = st.slider("Langsamer Abbau (Schutz)", 0.02, 0.06, 0.03, 0.01, key='k_langsam_direkt')
            
            sim_jahre = st.slider("Simulationszeit (Jahre)", 30, 100, 60, 10)
        
        with col2:
            f_schnell = 0.7
            k_schnell_beide = 1.5
            eps = 0.2
            
            def lstm_szenario(y, t, I, k_l):
                C_s, C_l = y
                dCs = f_schnell * I - k_schnell_beide * C_s
                dCl = (1-f_schnell) * I + eps * k_schnell_beide * C_s - k_l * C_l
                return [dCs, dCl]
            
            t = np.linspace(0, sim_jahre, 500)
            
            sol_konv = odeint(lstm_szenario, [0, 0], t, args=(input_konv, k_langsam_konv))
            C_gesamt_konv = sol_konv[:, 0] + sol_konv[:, 1]
            
            sol_direkt = odeint(lstm_szenario, [0, 0], t, args=(input_direkt, k_langsam_direkt))
            C_gesamt_direkt = sol_direkt[:, 0] + sol_direkt[:, 1]
            
            C_s_eq_konv = f_schnell * input_konv / k_schnell_beide
            C_l_eq_konv = ((1-f_schnell)*input_konv + eps*k_schnell_beide*C_s_eq_konv) / k_langsam_konv
            gesamt_eq_konv = C_s_eq_konv + C_l_eq_konv
            
            C_s_eq_direkt = f_schnell * input_direkt / k_schnell_beide
            C_l_eq_direkt = ((1-f_schnell)*input_direkt + eps*k_schnell_beide*C_s_eq_direkt) / k_langsam_direkt
            gesamt_eq_direkt = C_s_eq_direkt + C_l_eq_direkt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(t, C_gesamt_konv, 'r-', lw=3, label=f'Konventionell (→ {gesamt_eq_konv:.1f} Mg C/ha)')
            ax.plot(t, C_gesamt_direkt, 'g-', lw=3, label=f'Direktsaat (→ {gesamt_eq_direkt:.1f} Mg C/ha)')
            
            ax.fill_between(t, C_gesamt_konv, C_gesamt_direkt, alpha=0.2, color='green',
                           label=f'Kohlenstoffgewinn: {gesamt_eq_direkt - gesamt_eq_konv:.1f} Mg C/ha')
            
            ax.axhline(gesamt_eq_konv, ls='--', color='red', alpha=0.5)
            ax.axhline(gesamt_eq_direkt, ls='--', color='green', alpha=0.5)
            
            aktuell_konv = C_gesamt_konv[-1]
            aktuell_direkt = C_gesamt_direkt[-1]
            ax.scatter([sim_jahre, sim_jahre], [aktuell_konv, aktuell_direkt],
                      s=200, c=['red', 'green'], edgecolors='black', linewidths=2, zorder=10)
            
            ax.set_xlabel('Zeit (Jahre)', fontsize=12)
            ax.set_ylabel('Gesamt Bodenkohlenstoff (Mg C/ha)', fontsize=12)
            ax.set_title('Vergleich von Management-Praktiken', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='lower right')
            ax.grid(alpha=0.3)
            
            st.pyplot(fig)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>🌾 Landwirtschaftliche Implikationen:</strong>
        <ul>
        <li><strong>Nach {sim_jahre} Jahren:</strong>
            <ul>
            <li>Konventionell: {C_gesamt_konv[-1]:.1f} Mg C/ha</li>
            <li>Direktsaat: {C_gesamt_direkt[-1]:.1f} Mg C/ha</li>
            <li><strong>Differenz: {C_gesamt_direkt[-1] - C_gesamt_konv[-1]:.1f} Mg C/ha gespeichert</strong></li>
            </ul>
        </li>
        <li><strong>Im Gleichgewicht:</strong> {gesamt_eq_direkt - gesamt_eq_konv:.1f} Mg C/ha mehr mit Direktsaat</li>
        <li><strong>Warum?</strong> Höherer Input + niedrigere Abbaurate = beide Effekte verstärken sich!</li>
        <li><strong>Handlungsempfehlung:</strong> Wechsel zu Direktsaat + Zwischenfrüchte zum Bodenkohlenstoffaufbau!</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("🎯 Systemresilienz: Erholung vs. Kollaps")
        
        st.markdown("""
        <div class="context-box">
        <strong>🔥 Das Szenario:</strong>
        <p>Ein gut bewirtschafteter Betrieb mit 60 Mg C/ha Bodenkohlenstoff (akkumuliert über 50 Jahre).</p>
        <p><strong>Störung:</strong> Extremes Erosionsereignis entfernt 50% des Oberbodens (und Bodenkohlenstoff).</p>
        <p><strong>Frage:</strong> Bestimmt das Management die Erholung?</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Vor Störung:**")
            initial_praxis = st.radio("Initiales Management", ["Konventionell", "Direktsaat"])
            
            st.markdown("**Nach Störung (wähle Reaktion):**")
            reaktion_praxis = st.radio("Erholungs-Management",
                                      ["Konventionell beibehalten",
                                       "Zu Direktsaat wechseln",
                                       "Degradation verstärken"])
            
            stoerung_zeit = st.slider("Wann tritt Störung auf? (Jahr)", 30, 70, 50, 5)
            stoerung_schwere = st.slider("Kohlenstoffverlust (%)", 30, 70, 50, 5)
        
        with col2:
            if initial_praxis == "Konventionell":
                I_initial = 2.5
                k_l_initial = 0.08
            else:
                I_initial = 4.0
                k_l_initial = 0.03
            
            if reaktion_praxis == "Konventionell beibehalten":
                I_nach = 2.5
                k_l_nach = 0.08
                farbe = 'orange'
            elif reaktion_praxis == "Zu Direktsaat wechseln":
                I_nach = 4.5
                k_l_nach = 0.025
                farbe = 'green'
            else:
                I_nach = 1.5
                k_l_nach = 0.12
                farbe = 'red'
            
            f_s = 0.7
            k_s = 1.5
            eps = 0.2
            
            def lstm_stoerung(y, t, I, k_l):
                C_s, C_l = y
                dCs = f_s * I - k_s * C_s
                dCl = (1-f_s) * I + eps * k_s * C_s - k_l * C_l
                return [dCs, dCl]
            
            t1 = np.linspace(0, stoerung_zeit, 300)
            sol1 = odeint(lstm_stoerung, [0, 0], t1, args=(I_initial, k_l_initial))
            
            C_vor = sol1[-1, :]
            C_nach_stoerung = C_vor * (1 - stoerung_schwere/100)
            
            t2 = np.linspace(stoerung_zeit, 100, 300)
            sol2 = odeint(lstm_stoerung, C_nach_stoerung, t2, args=(I_nach, k_l_nach))
            
            t_voll = np.concatenate([t1, t2])
            C_schnell_voll = np.concatenate([sol1[:, 0], sol2[:, 0]])
            C_langsam_voll = np.concatenate([sol1[:, 1], sol2[:, 1]])
            C_gesamt_voll = C_schnell_voll + C_langsam_voll
            
            C_s_eq_nach = f_s * I_nach / k_s
            C_l_eq_nach = ((1-f_s)*I_nach + eps*k_s*C_s_eq_nach) / k_l_nach
            gesamt_eq_nach = C_s_eq_nach + C_l_eq_nach
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(t_voll, C_gesamt_voll, color=farbe, lw=3, label=reaktion_praxis)
            ax.axvline(stoerung_zeit, ls='--', color='black', lw=2, alpha=0.5, label='Störungsereignis')
            ax.axhline(gesamt_eq_nach, ls=':', color=farbe, alpha=0.7, label=f'Neues Gleichgewicht ({gesamt_eq_nach:.1f})')
            
            ax.axvspan(stoerung_zeit-1, stoerung_zeit+1, alpha=0.3, color='red')
            
            C_vor_gesamt = C_vor[0] + C_vor[1]
            C_nach_gesamt = C_nach_stoerung[0] + C_nach_stoerung[1]
            ax.annotate(f'{stoerung_schwere}% Verlust\n({C_vor_gesamt:.1f} → {C_nach_gesamt:.1f} Mg C/ha)',
                       xy=(stoerung_zeit, C_nach_gesamt),
                       xytext=(stoerung_zeit-15, C_nach_gesamt-5),
                       fontsize=10, ha='center',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
            
            ax.set_xlabel('Zeit (Jahre)', fontsize=12)
            ax.set_ylabel('Gesamt Bodenkohlenstoff (Mg C/ha)', fontsize=12)
            ax.set_title('Erholungstrajektorie nach Störung', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, max(C_gesamt_voll)*1.15)
            
            st.pyplot(fig)
        
        erholungszeit = 100 - stoerung_zeit
        C_ende = C_gesamt_voll[-1]
        prozent_erholt = (C_ende - C_nach_gesamt) / (gesamt_eq_nach - C_nach_gesamt) * 100 if gesamt_eq_nach > C_nach_gesamt else 0
        
        if reaktion_praxis == "Zu Direktsaat wechseln" and gesamt_eq_nach > C_vor_gesamt:
            st.markdown(f"""
            <div class="insight-box">
            <strong>✅ RESILIENTES SYSTEM - BESSER ALS VORHER!</strong>
            <ul>
            <li><strong>Vor Störung:</strong> {C_vor_gesamt:.1f} Mg C/ha</li>
            <li><strong>Nach Störung:</strong> {C_nach_gesamt:.1f} Mg C/ha ({stoerung_schwere}% Verlust)</li>
            <li><strong>Nach {erholungszeit} Jahren:</strong> {C_ende:.1f} Mg C/ha</li>
            <li><strong>Neues Gleichgewicht:</strong> {gesamt_eq_nach:.1f} Mg C/ha (ÜBERTRIFFT Original!)</li>
            </ul>
            <p><strong>🌟 Die simpleLSTM Lektion:</strong> Schnelle Reaktion (Rest-Management) + Langsame Akkumulation (Humusschutz)
            = System kann sich nicht nur erholen sondern VERBESSERN nach Störung!</p>
            </div>
            """, unsafe_allow_html=True)
        elif gesamt_eq_nach > C_nach_gesamt and prozent_erholt > 50:
            st.markdown(f"""
            <div class="insight-box">
            <strong>✅ ERHOLENDES SYSTEM</strong>
            <ul>
            <li><strong>Nach {erholungszeit} Jahren:</strong> {prozent_erholt:.0f}% erholt</li>
            <li><strong>Neues Gleichgewicht:</strong> {gesamt_eq_nach:.1f} Mg C/ha</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
            <strong>❌ DEGRADIERENDES SYSTEM - KOLLAPS-TRAJEKTORIE!</strong>
            <ul>
            <li><strong>Vor:</strong> {C_vor_gesamt:.1f} Mg C/ha</li>
            <li><strong>Nach Störung:</strong> {C_nach_gesamt:.1f} Mg C/ha</li>
            <li><strong>Neues Gleichgewicht:</strong> {gesamt_eq_nach:.1f} Mg C/ha (NIEDRIGER als nach Störung!)</li>
            </ul>
            <p><strong>⚠️ Kritische Erkenntnis:</strong> Mit degradierendem Management kann das System nicht mal
            Post-Störungs-Niveaus halten. Der langsame Pool nimmt weiter ab!</p>
            <p><strong>Das ist der RAND DES KOLLAPS</strong> - Intervention nötig!</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>🔄 Aber jetzt haben Systeme auch INTERAKTIONEN...</strong>
        <p>In echten Ökosystemen beeinflussen sich Prozesse gegenseitig:</p>
        <ul>
        <li>Mehr Vegetation → Mehr Boden-Input</li>
        <li>Besserer Boden → Unterstützt mehr Vegetation</li>
        </ul>
        <p><strong>Dies sind RÜCKKOPPLUNGEN - und sie können das Systemverhalten dominieren!</strong></p>
        <p><strong>Weiter:</strong> Wenn Rückkopplungen die Kontrolle übernehmen</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# SCHRITT 4: VEGETATION-BODEN RÜCKKOPPLUNG
# ============================================================================
elif "Schritt 4" in schritt:
    st.header("🔄 Schritt 4: Wenn Rückkopplungen die Kontrolle übernehmen")
    
    st.markdown("""
    <div class="context-box">
    <h3>🌿 Von Kohlenstoff zu Ökosystemen: Die Vegetation-Boden Schleife</h3>
    <p><strong>Jetzt verstehst du:</strong></p>
    <ol>
    <li>Wie Zusammenhänge zu finden sind (Regression)</li>
    <li>Wie Wachstum Grenzen erreicht (Logistik)</li>
    <li>Wie schnelle und langsame Prozesse interagieren (simpleLSTM)</li>
    </ol>
    <p><strong>Das letzte Puzzlestück:</strong> Was passiert, wenn diese Prozesse sich gegenseitig rückkoppeln?</p>
    <p><strong>Das System:</strong> Mehr Vegetation → Mehr Streu → Besserer Boden → Unterstützt mehr Vegetation (POSITIVE RÜCKKOPPLUNG!)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Systemparameter")
    st.sidebar.markdown("**🌱 Vegetationsdynamik:**")
    g = st.sidebar.slider("Wachstumsrate g", 0.2, 1.0, 0.5, 0.05)
    m = st.sidebar.slider("Mortalität m", 0.05, 0.3, 0.1, 0.05)
    st.sidebar.markdown("**🌍 Bodendynamik:**")
    l = st.sidebar.slider("Streuproduktion l", 0.1, 0.5, 0.2, 0.05)
    d = st.sidebar.slider("OM-Abbau d", 0.02, 0.15, 0.05, 0.01)
    st.sidebar.markdown("**🔗 Rückkopplungsstärke:**")
    K_min = st.sidebar.slider("Min Tragfähigkeit", 2, 8, 5, 1)
    K_max = st.sidebar.slider("Max Tragfähigkeit", 12, 25, 20, 1)
    S_halb = st.sidebar.slider("Halb-Sättigung", 5, 20, 10, 1)
    
    tab1, tab2 = st.tabs(["🔄 Die Rückkopplungsschleife", "🌍 Alternative Zustände"])
    
    with tab1:
        st.subheader("Die Rückkopplung verstehen: Positiv → Negativ")
        st.markdown("**💡 Währungen: STOFF (C in Boden) + ORGANISATION (V) | Werkzeug: Wechselwirkungen & Rückkopplungen**")
        
        st.markdown("""
        **Das gekoppelte System mit Allee-Effekt:**
        
        ```
        dV/dt = g × V × (V/V_krit - 1) × (1 - V/K(S)) - m × V
                ↑        ↑ ALLEE-SCHWELLE  ↑ TRAGFÄHIGKEIT  ↑ Mortalität
                         (positiv→negativ)
        
        dS/dt = l × V - d × S
                ↑ Input   ↑ Abbau
        ```
        
        **Die Rückkopplung:** 
        ```
        K(S) = K_min + (K_max - K_min) × S/(S_halb + S)
        ```
        
        **Variablen & ihre Bedeutung:**
        - `V` = Vegetationsbiomasse [kg/m²] (ORGANISATION)
        - `S` = Boden-Organische-Substanz [kg/m²] (STOFF-Speicher)
        - `V_krit` = Kritische Vegetationsschwelle [kg/m²] (Allee-Effekt!)
        - `K(S)` = Tragfähigkeit als Funktion von Boden [kg/m²]
        - `g` = Max. Wachstumsrate [1/Jahr]
        - `m` = Mortalitätsrate [1/Jahr]
        - `l` = Litter-Produktionsrate [dimensionslos]
        - `d` = Boden-OM Abbaurate [1/Jahr]
        
        **Der Allee-Effekt:**
        - **V < V_krit:** POSITIVE Rückkopplung nach UNTEN (Degradation beschleunigt!)
        - **V ≈ V_krit:** Kipppunkt (instabil)
        - **V > V_krit:** NEGATIVE Rückkopplung (Selbststabilisierung)
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**🎮 Anfangsbedingungen testen:**")
            V0_test = st.slider("Anfangs-Vegetation V₀ (kg/m²)", 1, 20, 8, 1)
            S0_test = st.slider("Anfangs-Boden OM S₀ (kg/m²)", 5, 50, 20, 2)
            
            K_aktuell = K_min + (K_max - K_min) * S0_test/(S_halb + S0_test)
            st.metric("Aktuelle Tragfähigkeit K(S)", f"{K_aktuell:.1f} kg/m²")
            
            # Calculate Allee threshold from sidebar params
            V_krit = K_min * 0.3  # 30% of min carrying capacity
            st.markdown("---")
            st.metric("🔴 Allee-Schwelle V_krit", f"{V_krit:.1f} kg/m²",
                     help="Unter dieser Schwelle: positive Rückkopplung → Degradation!")
            
            if V0_test < V_krit:
                st.error("⚠️ START UNTER SCHWELLE → Degradation erwartet!")
            elif V0_test < K_aktuell * 0.7:
                st.warning("⚙️ Zwischen Schwelle und Tragfähigkeit → Wachstum möglich")
            else:
                st.success("✅ Nahe Tragfähigkeit → Stabil")
        
        with col2:
            def veg_boden_system(y, t):
                V, S = y
                K = K_min + (K_max - K_min) * S/(S_halb + S)
                
                # Allee-Effekt: Wachstum negativ unter Schwelle V_krit
                V_krit = K_min * 0.3  # 30% von minimaler Tragfähigkeit
                allee_faktor = (V / V_krit - 1)  # negativ wenn V < V_krit!
                
                dV = g * V * allee_faktor * (1 - V/K) - m*V
                dS = l*V - d*S
                return [dV, dS]
            
            t = np.linspace(0, 200, 1500)
            sol = odeint(veg_boden_system, [V0_test, S0_test], t)
            
            # FIXE ACHSEN
            V_max_axis = K_max * 1.1
            S_max_axis = 60
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 11))
            
            # Panel 1: Zeitreihen mit fixen Achsen
            ax = axes[0, 0]
            ax.plot(t, sol[:, 0], 'g-', lw=3, label='Vegetation V (ORGANISATION)')
            ax.plot(t, sol[:, 1], 'brown', lw=3, label='Boden-OM S (STOFF)')
            
            # Markiere Allee-Schwelle
            V_krit = K_min * 0.3
            ax.axhline(V_krit, ls='--', color='red', lw=2, alpha=0.7, 
                      label=f'Allee-Schwelle V_krit={V_krit:.1f}')
            ax.fill_between(t, 0, V_krit, alpha=0.15, color='red', label='Degradationszone')
            
            ax.set_xlabel('Zeit (Jahre)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Zustandsvariablen', fontsize=12, fontweight='bold')
            ax.set_title('Zeitliche Entwicklung mit Allee-Effekt', fontweight='bold', fontsize=13)
            ax.legend(fontsize=9, loc='best')
            ax.grid(alpha=0.3)
            ax.set_ylim(0, max(V_max_axis, S_max_axis))
            ax.set_xlim(0, 200)
            
            # Panel 2: Phasenebene mit Nullklinien
            ax = axes[0, 1]
            ax.plot(sol[:, 1], sol[:, 0], 'purple', lw=3, alpha=0.8, label='Trajektorie')
            ax.scatter([S0_test], [V0_test], s=250, c='green', marker='o',
                      edgecolor='black', linewidth=2.5, zorder=10, label='Start')
            ax.scatter([sol[-1, 1]], [sol[-1, 0]], s=350, c='red', marker='*',
                      edgecolor='black', linewidth=2.5, zorder=10, label='Ende')
            
            # S-Nullklinie: dS/dt = 0 → V = d*S/l
            S_bereich = np.linspace(0, S_max_axis, 100)
            V_S_null = d * S_bereich / l
            ax.plot(S_bereich, V_S_null, 'b--', lw=2, alpha=0.7, label='S-Nullklinie (dS/dt=0)')
            
            # Allee-Schwelle horizontal
            ax.axhline(V_krit, ls='--', color='red', lw=2, alpha=0.7, label=f'Allee-Schwelle')
            ax.fill_between([0, S_max_axis], 0, V_krit, alpha=0.1, color='red')
            
            ax.set_xlabel('Boden-OM S (kg/m²)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Vegetation V (kg/m²)', fontsize=12, fontweight='bold')
            ax.set_title('Phasenebene: Positive ↔ Negative Rückkopplung', fontweight='bold', fontsize=13)
            ax.legend(fontsize=9, loc='best')
            ax.grid(alpha=0.3)
            ax.set_xlim(0, S_max_axis)
            ax.set_ylim(0, V_max_axis)
            
            # Panel 3: Tragfähigkeit als Funktion von Boden
            ax = axes[1, 0]
            S_K = np.linspace(0, S_max_axis, 200)
            K_werte = K_min + (K_max - K_min) * S_K/(S_halb + S_K)
            ax.plot(S_K, K_werte, 'orange', lw=4, label='K(S) - Boden-abhängig')
            ax.axhline(K_min, ls=':', color='red', lw=2, label=f'K_min = {K_min}')
            ax.axhline(K_max, ls=':', color='green', lw=2, label=f'K_max = {K_max}')
            ax.axhline(V_krit, ls='--', color='darkred', lw=2, label=f'V_krit = {V_krit:.1f}')
            ax.axvline(S_halb, ls=':', color='gray', lw=2, label=f'S_halb = {S_halb}')
            
            # Aktueller Zustand
            if len(sol) > 0:
                ax.scatter([sol[-1, 1]], [K_min + (K_max - K_min) * sol[-1,1]/(S_halb + sol[-1,1])],
                          s=250, c='blue', edgecolor='black', linewidth=2, zorder=10, label='Aktueller Zustand')
            
            ax.set_xlabel('Boden-OM S (kg/m²)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Tragfähigkeit K (kg/m²)', fontsize=12, fontweight='bold')
            ax.set_title('Die Rückkopplungsfunktion K(S)', fontweight='bold', fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlim(0, S_max_axis)
            ax.set_ylim(0, V_max_axis)
            
            # Panel 4: Wachstumsrate als Funktion von V (zeigt Allee-Effekt!)
            ax = axes[1, 1]
            V_range = np.linspace(0.1, V_max_axis, 300)
            # Berechne Wachstumsrate bei mittlerem Boden
            S_mittel = S0_test
            K_mittel = K_min + (K_max - K_min) * S_mittel/(S_halb + S_mittel)
            allee_faktoren = (V_range / V_krit - 1)
            wachstumsraten = g * V_range * allee_faktoren * (1 - V_range/K_mittel) - m*V_range
            
            ax.plot(V_range, wachstumsraten, 'purple', lw=4)
            ax.axhline(0, color='black', lw=1.5)
            ax.axvline(V_krit, ls='--', color='red', lw=2, alpha=0.7, label='Allee-Schwelle')
            
            # Markiere Zonen
            ax.fill_between(V_range, 0, wachstumsraten, where=(wachstumsraten<0), alpha=0.2, 
                           color='red', label='dV/dt < 0 (Degradation)')
            ax.fill_between(V_range, 0, wachstumsraten, where=(wachstumsraten>0), alpha=0.2,
                           color='green', label='dV/dt > 0 (Wachstum)')
            
            # Finde Gleichgewichte
            sign_changes = np.where(np.diff(np.sign(wachstumsraten)))[0]
            for idx in sign_changes:
                if idx > 0 and idx < len(V_range)-1:
                    stabil = wachstumsraten[idx-1] > wachstumsraten[idx+1]
                    ax.scatter([V_range[idx]], [0], s=300, 
                             c='green' if stabil else 'red',
                             marker='o' if stabil else 'x',
                             edgecolor='black', linewidths=2, zorder=10)
            
            ax.set_xlabel('Vegetation V (kg/m²)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Wachstumsrate dV/dt', fontsize=12, fontweight='bold')
            ax.set_title('Allee-Effekt: Positive Rückkopplung bei V < V_krit!', 
                        fontweight='bold', fontsize=13, color='darkred')
            ax.legend(fontsize=9, loc='best')
            ax.grid(alpha=0.3)
            ax.set_xlim(0, V_max_axis)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>🔄 Der Allee-Effekt (Positive → Negative Rückkopplung):</strong>
        <br><br>
        <strong>Drei Bereiche:</strong>
        <ol>
        <li><strong>V < {V_krit:.1f} kg/m² (Unter Allee-Schwelle):</strong>
            <ul>
            <li>🔴 POSITIVE Rückkopplung nach UNTEN!</li>
            <li>Zu wenig Vegetation → Bodenerosion → Noch weniger Vegetation</li>
            <li>Selbstverstärkende Degradation → Kollaps!</li>
            </ul>
        </li>
        <li><strong>V ≈ {V_krit:.1f} kg/m² (An der Schwelle):</strong>
            <ul>
            <li>⚠️ KIPPPUNKT - instabiles Gleichgewicht</li>
            <li>Kleine Störung entscheidet: Erholung oder Kollaps</li>
            </ul>
        </li>
        <li><strong>V > {V_krit:.1f} kg/m² (Über Schwelle):</strong>
            <ul>
            <li>✅ NEGATIVE Rückkopplung (wie in Schritt 2!)</li>
            <li>Mehr Vegetation → Mehr Boden → Höheres K → Stabilisierung</li>
            <li>Selbstregulierung zum Gleichgewicht</li>
            </ul>
        </li>
        </ol>
        <br>
        <strong>🔗 Werkzeug-Verbindung:</strong> Dies ist <em>Werkzeug #3: Wechselwirkungen & Rückkopplungen</em> - 
        Sowohl positive (Allee) als auch negative (logistisch) Rückkopplungen im selben System!
        <br><br>
        <strong>Management-Implikation:</strong> Degradierte Systeme (V < V_krit) können sich NICHT selbst erholen - 
        sie brauchen massive Intervention um über die Schwelle zu kommen!
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>🔄 Der Rückkopplungsmechanismus:</strong>
        <ol>
        <li><strong>Mehr Vegetation</strong> (V ↑) → Mehr Streuproduktion (l×V ↑)</li>
        <li><strong>Mehr Streu</strong> → Boden OM akkumuliert (S ↑)</li>
        <li><strong>Besserer Boden</strong> (S ↑) → Höhere Tragfähigkeit (K(S) ↑)</li>
        <li><strong>Höheres K</strong> → Unterstützt MEHR VEGETATION (V ↑)</li>
        <li><strong>Schleife geht weiter...</strong> (POSITIVE RÜCKKOPPLUNG!)</li>
        </ol>
        <p><strong>Ergebnis:</strong> System verstärkt Anfangsbedingungen - kann zu alternativen stabilen Zuständen führen!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("🌍 Alternative stabile Zustände: Wald vs. Grasland")
        
        st.markdown("""
        <div class="context-box">
        <strong>🌲🌾 Die Savannenfrage:</strong>
        <p>Gleiches Klima, gleicher Niederschlag - aber manche Gebiete sind Wald, andere Grasland. Warum?</p>
        <p><strong>Antwort:</strong> Bistabilität! Die positive Rückkopplung erzeugt ZWEI stabile Zustände.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_trajektorien = st.slider("Anzahl Trajektorien", 3, 8, 5, 1)
        
        with col2:
            np.random.seed(42)
            V0_liste = np.linspace(2, K_max*0.9, n_trajektorien)
            S0_liste = np.linspace(5, 35, n_trajektorien)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            farben = plt.cm.viridis(np.linspace(0, 1, n_trajektorien))
            
            t = np.linspace(0, 200, 800)
            
            for i, (V0, S0, farbe) in enumerate(zip(V0_liste, S0_liste, farben)):
                sol = odeint(veg_boden_system, [V0, S0], t)
                
                ax1.plot(sol[:, 1], sol[:, 0], color=farbe, lw=2, alpha=0.7)
                ax1.scatter([S0], [V0], s=100, c=[farbe], edgecolor='black', linewidth=1.5, zorder=10)
                ax1.scatter([sol[-1, 1]], [sol[-1, 0]], s=200, marker='*', c=[farbe],
                           edgecolor='black', linewidth=2, zorder=10)
                
                ax2.plot(t, sol[:, 0], color=farbe, lw=1.5, alpha=0.7,
                        label=f'Start: V={V0:.1f}, S={S0:.1f}')
            
            S_bereich = np.linspace(0, 50, 100)
            V_S_null = d * S_bereich / l
            ax1.plot(S_bereich, V_S_null, 'b--', lw=2, alpha=0.5, label='S-Nullklinie')
            
            ax1.set_xlabel('Boden OM S', fontsize=12)
            ax1.set_ylabel('Vegetation V', fontsize=12)
            ax1.set_title('Phasenebene: Konvergieren sie?', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)
            
            ax2.set_xlabel('Zeit (Jahre)', fontsize=12)
            ax2.set_ylabel('Vegetationsbiomasse', fontsize=12)
            ax2.set_title('Zeitreihen: Verschiedene Trajektorien', fontsize=14, fontweight='bold')
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        endzustaende = []
        for V0, S0 in zip(V0_liste, S0_liste):
            sol = odeint(veg_boden_system, [V0, S0], np.linspace(0, 200, 500))
            endzustaende.append((sol[-1, 0], sol[-1, 1]))
        
        end_V = [v for v, s in endzustaende]
        end_S = [s for v, s in endzustaende]
        
        if max(end_V) - min(end_V) > 5:
            st.markdown(f"""
            <div class="warning-box">
            <strong>⚠️ BISTABILITÄT ERKANNT!</strong>
            <p><strong>Verschiedene Anfangsbedingungen führen zu verschiedenen Endzuständen:</strong></p>
            <ul>
            <li><strong>Degradierter Zustand:</strong> V ≈ {min(end_V):.1f}, S ≈ {min(end_S):.1f}</li>
            <li><strong>Gesunder Zustand:</strong> V ≈ {max(end_V):.1f}, S ≈ {max(end_S):.1f}</li>
            </ul>
            <p><strong>Das bedeutet:</strong> Geschichte ist wichtig! Wo du startest bestimmt wo du endest.</p>
            <p><strong>Für Management:</strong> Degradierte Systeme brauchen einen GROSSEN Schub um zum gesunden Zustand zu kippen!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box">
            <strong>✅ Einzelner stabiler Zustand</strong>
            <p>Alle Trajektorien konvergieren zu ähnlichem Gleichgewicht.</p>
            <p><strong>Versuche:</strong> Rückkopplungsstärke erhöhen um Bistabilität zu erzeugen!</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>🦊 Aber Rückkopplungen sind nicht immer zwischen Vegetation und Boden...</strong>
        <p>In der Natur gibt es viele Arten von Interaktionen:</p>
        <ul>
        <li><strong>Räuber-Beute:</strong> Fuchs jagt Kaninchen, Kaninchen ernähren Füchse</li>
        <li><strong>Konkurrenz:</strong> Pflanzen um Licht, Wasser, Nährstoffe</li>
        <li><strong>Mutualismus:</strong> Mykorrhiza-Pilze und Baumwurzeln</li>
        </ul>
        <p><strong>Diese Interaktionen folgen ähnlichen Prinzipien - aber mit verschiedenen Zeitskalen und Kopplungen!</strong></p>
        <p><strong>Weiter:</strong> Klassische Räuber-Beute Dynamik verstehen</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# SCHRITT 5: RÄUBER-BEUTE
# ============================================================================
elif "Schritt 5" in schritt:
    st.header("🦊 Schritt 5: Interaktionen & Koexistenz (Räuber-Beute)")
    
    st.markdown("""
    <div class="context-box">
    <h3>🦊🐇 Von Rückkopplungen zu Interaktionen</h3>
    <p><strong>Du hast gelernt:</strong></p>
    <ol>
    <li>Stock-Flow Denken (Akkumulation mit Grenzen)</li>
    <li>simpleLSTM (Schnelle & langsame Zeitskalen)</li>
    <li>Positive Rückkopplungen (Vegetation-Boden)</li>
    </ol>
    <p><strong>Jetzt:</strong> Was passiert wenn zwei <em>Populationen</em> interagieren?</p>
    <p><strong>Das klassische Beispiel:</strong> Füchse (Räuber) jagen Kaninchen (Beute)</p>
    <p><strong>Die Frage:</strong> Können sie koexistieren? Oder stirbt einer aus?</p>
    <p><strong>Die Erkenntnis:</strong> Interaktionen + verschiedene Zeitskalen → Oszillationen!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🔬 Das Konzept", "📊 Rosenzweig-MacArthur Modell", "🌍 Biotisch-Abiotische Kopplung"])
    
    with tab1:
        st.subheader("Räuber-Beute Grundlagen")
        
        st.markdown("""
        <div class="context-box">
        <strong>🦊🐇 Die Dynamik:</strong>
        <ol>
        <li>Viele Kaninchen → Füchse haben viel Nahrung → Fuchspopulation wächst</li>
        <li>Viele Füchse → Starke Prädation → Kaninchenpopulation sinkt</li>
        <li>Wenige Kaninchen → Füchse verhungern → Fuchspopulation sinkt</li>
        <li>Wenige Füchse → Kaninchen können sich erholen → Kaninchenpopulation wächst</li>
        <li>... und der Zyklus beginnt von vorne!</li>
        </ol>
        <p><strong>Dies ist eine OSZILLATION - kein Gleichgewicht!</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**📋 Vereinfachtes Modell (nur zum Konzept):**")
            st.markdown("""
            **Kaninchen (Beute):**
            - Wachstum: +r×N
            - Verlust durch Prädation: -a×N×P
            
            **Füchse (Räuber):**
            - Wachstum durch Fressen: +e×a×N×P
            - Mortalität: -m×P
            
            Wobei:
            - N = Kaninchen
            - P = Füchse
            - r = Kaninchen Wachstumsrate
            - a = Angriffsrate
            - e = Umwandlungseffizienz
            - m = Fuchs Mortalität
            """)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <strong>🔗 Verbindung zu vorherigen Konzepten:</strong>
            
            <p><strong>1. Stock-Flow (Schritt 2):</strong></p>
            <ul>
            <li>N und P sind Bestände (Populationen)</li>
            <li>Geburten und Tode sind Flüsse</li>
            </ul>
            
            <p><strong>2. simpleLSTM (Schritt 3):</strong></p>
            <ul>
            <li>Kaninchen = schneller (kürzere Generation)</li>
            <li>Füchse = langsamer (längere Generation)</li>
            <li>Zeitverzögerung erzeugt Oszillationen!</li>
            </ul>
            
            <p><strong>3. Rückkopplung (Schritt 4):</strong></p>
            <ul>
            <li>Mehr N → Mehr P (positiv für Räuber)</li>
            <li>Mehr P → Weniger N (negativ für Beute)</li>
            <li>Gegenseitige Regelung!</li>
            </ul>
            
            <p><strong>4. Neue Elemente:</strong></p>
            <ul>
            <li><strong>Sättigung:</strong> Füchse können nicht unbegrenzt fressen</li>
            <li><strong>Energietransfer:</strong> Nur Bruchteil e wird zu Fuchsbiomasse</li>
            <li><strong>Zwei interagierende Arten:</strong> Nicht nur eine Variable wie vorher</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Problem mit einfachem Modell:</strong>
        <p>Das obige Modell (Lotka-Volterra) erzeugt <strong>neutrale Zyklen</strong> - 
        Amplitude hängt von Anfangsbedingungen ab, ändert sich aber nie.</p>
        <p><strong>Unrealistisch!</strong> In der Natur gibt es Dämpfung oder Verstärkung.</p>
        <p><strong>Lösung:</strong> Realistische Erweiterungen hinzufügen...</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Das Rosenzweig-MacArthur Modell")
        
        st.markdown("""
        <div class="context-box">
        <strong>📚 Verbesserungen zum einfachen Modell:</strong>
        <ol>
        <li><strong>Beute-Selbstlimitierung:</strong> Kaninchen haben Tragfähigkeit K (wie in Schritt 2!)</li>
        <li><strong>Funktionelle Reaktion Typ II:</strong> Füchse sättigen (können nicht unbegrenzt fressen)</li>
        <li><strong>Ergebnis:</strong> Realistische Konvergenz zu stabilem Gleichgewicht oder Grenzzyklen</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**🎚️ Parameter:**")
            
            st.markdown("**🐇 Kaninchen (Beute):**")
            r = st.slider("Wachstumsrate r (1/Jahr)", 0.5, 2.0, 1.0, 0.1)
            K = st.slider("Tragfähigkeit K", 50, 200, 100, 10)
            
            st.markdown("**🦊 Füchse (Räuber):**")
            a = st.slider("Angriffsrate a", 1.0, 10.0, 5.0, 0.5)
            h = st.slider("Handhabungszeit h", 5, 50, 20, 5)
            e = st.slider("Umwandlungseffizienz e", 0.1, 0.5, 0.3, 0.05)
            m = st.slider("Mortalität m (1/Jahr)", 0.1, 1.0, 0.4, 0.1)
            
            st.markdown("**🎮 Anfangsbedingungen:**")
            N0 = st.slider("Anfangs-Kaninchen N₀", 10, 100, 50, 10)
            P0 = st.slider("Anfangs-Füchse P₀", 1, 20, 5, 1)
            
            sim_zeit = st.slider("Simulationszeit (Jahre)", 50, 200, 100, 10)
            
            st.markdown(f"""
            <div class="context-box">
            <strong>💡 Interpretation:</strong>
            <p><strong>Energieeffizienz:</strong> {e*100:.0f}%</p>
            <p>Nur {e*100:.0f}% der gefressenen Kaninchen-Biomasse wird zu Fuchs-Biomasse!</p>
            <p><strong>Dies ist ENERGIEWÄHRUNG!</strong> (Schritt 1 Verbindung)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            def rauber_beute(y, t):
                N, P = y
                dN = r*N*(1 - N/K) - (a*N)/(1 + a*h*N) * P
                dP = e*(a*N)/(1 + a*h*N) * P - m*P
                return [dN, dP]
            
            t = np.linspace(0, sim_zeit, 1000)
            sol = odeint(rauber_beute, [N0, P0], t)
            
            # Prüfe auf negative Werte
            if np.any(sol < 0):
                st.warning("⚠️ Population ging negativ - Parameter anpassen!")
                sol = np.maximum(sol, 0)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Zeitreihen
            ax1.plot(t, sol[:, 0], 'b-', lw=2.5, label='Kaninchen (schneller)', alpha=0.8)
            ax1.plot(t, sol[:, 1], 'r-', lw=2.5, label='Füchse (langsamer)', alpha=0.8)
            ax1.set_xlabel('Zeit (Jahre)', fontsize=12)
            ax1.set_ylabel('Population', fontsize=12)
            ax1.set_title('Zeitreihen: Gekoppelte Oszillationen', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11, loc='upper right')
            ax1.grid(alpha=0.3)
            
            # Finde Spitzen für Anmerkung
            from scipy.signal import find_peaks
            beute_spitzen, _ = find_peaks(sol[:, 0], distance=20)
            rauber_spitzen, _ = find_peaks(sol[:, 1], distance=20)
            
            if len(beute_spitzen) > 0:
                ax1.scatter(t[beute_spitzen], sol[beute_spitzen, 0], c='blue', s=80, zorder=10, alpha=0.5)
            if len(rauber_spitzen) > 0:
                ax1.scatter(t[rauber_spitzen], sol[rauber_spitzen, 1], c='red', s=80, zorder=10, alpha=0.5)
            
            # Phasenebene
            ax2.plot(sol[:, 0], sol[:, 1], 'purple', lw=2.5, alpha=0.7)
            ax2.scatter([N0], [P0], s=200, c='green', marker='o',
                       edgecolor='black', linewidth=2, zorder=10, label='Start')
            ax2.scatter([sol[-1, 0]], [sol[-1, 1]], s=300, c='red', marker='*',
                       edgecolor='black', linewidth=2, zorder=10, label='Ende')
            
            ax2.set_xlabel('Kaninchen (Beute)', fontsize=12)
            ax2.set_ylabel('Füchse (Räuber)', fontsize=12)
            ax2.set_title('Phasenebene: Zyklisches Verhalten', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(alpha=0.3)
            
            # Richtungspfeile hinzufügen
            if len(sol) > 50:
                pfeil_indizes = [int(len(sol)*f) for f in [0.1, 0.3, 0.5, 0.7, 0.9]]
                for idx in pfeil_indizes:
                    if idx < len(sol)-1:
                        ax2.annotate('', xy=(sol[idx+1, 0], sol[idx+1, 1]),
                                    xytext=(sol[idx, 0], sol[idx, 1]),
                                    arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("""
        <div class="insight-box">
        <strong>🔍 Räuber-Beute Interpretation:</strong>
        <ul>
        <li><strong>Gekoppelte Dynamik:</strong> Beute oszilliert, Räuber folgt mit Verzögerung</li>
        <li><strong>simpleLSTM:</strong> Verschiedene Generationszeiten erzeugen Phasenverzögerung</li>
        <li><strong>Energietransfer:</strong> Nur {:.0f}% effizient (Räuber brauchen {:.1f} Einheiten Beute für 1 Einheit Wachstum)</li>
        <li><strong>Trophische Ebene:</strong> Verknüpft ORGANISATION (Populationen) und ENERGIE (Transfereffizienz)</li>
        <li><strong>Sättigung:</strong> Typ II funktionelle Reaktion stabilisiert (verhindert unbegrenzte Prädation)</li>
        </ul>
        </div>
        """.format(e*100, 1/e), unsafe_allow_html=True)
        
        # Stabilitätsanalyse
        if sol[-1, 0] < 0.01 or sol[-1, 1] < 0.01:
            st.warning("⚠️ Eine Population ging aus! Parameter anpassen.")
        else:
            # Prüfe ob Oszillationen abklingen
            if len(beute_spitzen) > 2:
                amplituden = sol[beute_spitzen, 0]
                if len(amplituden) > 2 and amplituden[-1] < amplituden[0] * 0.9:
                    st.success(f"""
                    ✅ **Dämpfende Oszillationen** → Konvergiert zu stabilem Gleichgewicht
                    
                    Endzustand: ~{sol[-1, 0]:.0f} Kaninchen, ~{sol[-1, 1]:.0f} Füchse
                    """)
                else:
                    st.info("""
                    🔄 **Anhaltende Oszillationen** (Grenzzyklus)
                    
                    Populationen zyklieren weiter - typisch für mittlere Parameter!
                    """)
    
    with tab3:
        st.subheader("🌍 Biotisch-Abiotische Kopplung")
        
        st.markdown("""
        <div class="context-box">
        <strong>🌿💧 Alternative Perspektive: Pflanze-Wasser Interaktion</strong>
        <p>Räuber-Beute Prinzipien gelten auch für <strong>biotisch-abiotische</strong> Interaktionen!</p>
        <p><strong>Das System:</strong></p>
        <ul>
        <li><strong>Pflanzen (biotisch):</strong> Wachsen mit Wasser</li>
        <li><strong>Bodenfeuchte (abiotisch):</strong> Verbraucht durch Pflanzen, aufgefüllt durch Regen</li>
        </ul>
        <p><strong>Analog zu Räuber-Beute:</strong></p>
        <ul>
        <li>Pflanzen = "Räuber" (konsumieren Wasser)</li>
        <li>Wasser = "Beute" (limitierte Ressource)</li>
        <li>Regen = "Beute-Wachstum" (Nachschub)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**🎚️ Parameter:**")
            
            st.markdown("**💧 Wasser:**")
            P_regen = st.slider("Niederschlag P (mm/Monat)", 20, 150, 80, 10)
            
            st.markdown("**🌱 Pflanzen:**")
            r_pflanze = st.slider("Wachstumsrate (1/Monat)", 0.1, 0.8, 0.3, 0.05)
            m_pflanze = st.slider("Mortalität (1/Monat)", 0.05, 0.3, 0.1, 0.05)
            K_bio = st.slider("Max Biomasse", 50, 200, 100, 10)
            
            st.markdown("**💧🌱 Kopplung:**")
            ET_koeff = st.slider("Transpirationskoeffizient", 0.5, 3.0, 1.5, 0.1)
            wasser_schwelle = st.slider("Wasser-Stressschwelle (mm)", 20, 100, 50, 10)
            
            B0 = st.slider("Anfangs-Biomasse", 10, 80, 40, 10)
            W0 = st.slider("Anfangs-Bodenfeuchte (mm)", 30, 150, 80, 10)
        
        with col2:
            def pflanze_wasser(y, t):
                B, W = y  # Biomasse, Wasser
                
                # Wasser-Stress-Faktor
                wasser_faktor = min(1.0, W / wasser_schwelle)
                
                # Pflanzen-Wachstum (begrenzt durch Wasser UND Tragfähigkeit)
                dB = r_pflanze * B * (1 - B/K_bio) * wasser_faktor - m_pflanze * B
                
                # Wasser-Dynamik (Regen - ET)
                ET = ET_koeff * B * wasser_faktor
                dW = P_regen - ET
                
                return [dB, dW]
            
            t = np.linspace(0, 120, 1000)  # 10 Jahre in Monaten
            sol = odeint(pflanze_wasser, [B0, W0], t)
            
            # Verhindere negative Werte
            sol = np.maximum(sol, 0)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Zeitreihen
            ax1.plot(t/12, sol[:, 0], 'g-', lw=2.5, label='Pflanzen-Biomasse', alpha=0.8)
            ax1_twin = ax1.twinx()
            ax1_twin.plot(t/12, sol[:, 1], 'b-', lw=2.5, label='Bodenfeuchte', alpha=0.8)
            
            ax1.set_xlabel('Zeit (Jahre)', fontsize=12)
            ax1.set_ylabel('Biomasse', fontsize=12, color='g')
            ax1_twin.set_ylabel('Bodenfeuchte (mm)', fontsize=12, color='b')
            ax1.set_title('Biotisch-Abiotische Kopplung', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='g')
            ax1_twin.tick_params(axis='y', labelcolor='b')
            ax1.grid(alpha=0.3)
            
            # Legende kombinieren
            linien1, labels1 = ax1.get_legend_handles_labels()
            linien2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(linien1 + linien2, labels1 + labels2, loc='upper right')
            
            # Phasenebene
            ax2.plot(sol[:, 1], sol[:, 0], 'purple', lw=2.5, alpha=0.7)
            ax2.scatter([W0], [B0], s=200, c='green', marker='o',
                       edgecolor='black', linewidth=2, zorder=10, label='Start')
            ax2.scatter([sol[-1, 1]], [sol[-1, 0]], s=300, c='red', marker='*',
                       edgecolor='black', linewidth=2, zorder=10, label='Ende')
            
            ax2.axvline(wasser_schwelle, ls='--', color='blue', alpha=0.5, label='Wasser-Stressschwelle')
            
            ax2.set_xlabel('Bodenfeuchte (mm)', fontsize=12)
            ax2.set_ylabel('Pflanzen-Biomasse', fontsize=12)
            ax2.set_title('Phasenraum: Wasser-Vegetation Dynamik', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>🌿💧 Biotisch-Abiotische Interaktion:</strong>
        <ul>
        <li><strong>Gegenseitige Limitierung:</strong>
            <ul>
            <li>Wenig Wasser → Pflanzen können nicht wachsen (abiotisch limitiert biotisch)</li>
            <li>Viel Pflanzen → Hohe Transpiration → Wasser sinkt (biotisch beeinflusst abiotisch)</li>
            </ul>
        </li>
        <li><strong>Schwellenwert:</strong> Bei W < {wasser_schwelle} mm beginnt Wasser-Stress</li>
        <li><strong>Gleichgewicht möglich:</strong> Wenn P = ET (Niederschlag = Evapotranspiration)</li>
        <li><strong>Verbindung zu früheren Konzepten:</strong>
            <ul>
            <li>Schritt 2: Logistisches Wachstum (K_bio)</li>
            <li>Schritt 3: simpleLSTM (Wasser schneller, Biomasse langsamer)</li>
            <li>Schritt 4: Rückkopplung (Biomasse → Wasser → Biomasse)</li>
            </ul>
        </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="context-box">
        <strong>🎓 Die komplette Modellierungsreise:</strong>
        <p>Du hast jetzt gesehen wie:</p>
        <ol>
        <li><strong>Regression</strong> Muster in Daten findet</li>
        <li><strong>Stock-Flow → Logistik</strong> Akkumulation mit Grenzen beschreibt</li>
        <li><strong>simpleLSTM</strong> schnelle und langsame Prozesse trennt</li>
        <li><strong>Rückkopplungen</strong> alternative Zustände erzeugen können</li>
        <li><strong>Interaktionen</strong> (Räuber-Beute, Pflanze-Wasser) Oszillationen und Koexistenz erzeugen</li>
        </ol>
        <p><strong>Alle diese Konzepte sind verbunden - unterschiedliche Perspektiven auf dasselbe fundamentale Prinzip:</strong></p>
        <p><em>Ökosysteme sind dynamische Systeme mit Beständen, Flüssen, Rückkopplungen und mehreren Zeitskalen.</em></p>
        <p><strong>Du bist jetzt ein Modellierer! 🎉</strong></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**TU Bergakademie Freiberg**  
Geoökologische Grundlagen  
(cc) Conrad Jackisch & Maximilian Lau
Winter 2025/26
*coding mit Claude Sonnet 4.5*

**🎯 Lernziele:**
1. Muster erkennen (Regression)
2. Prozesse verstehen (Stock-Flow)
3. Zeitskalen sehen (simpleLSTM)
4. Rückkopplungen vorhersagen (Veg-Boden)
5. Interaktionen analysieren (Räuber-Beute)
""")
