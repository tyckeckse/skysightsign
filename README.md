# 🌤️ Skysightsign

**Live:** https://skysightsign.vercel.app · **Presentation:** [presentation/index.html](presentation/index.html)

En modell tränar på molntäcket i procent från gamla prognoser och lär sig det systematiska felet mellan dessa och historiska utfall. Eftersom molntäcket ofta är 0 eller 100 % (helt klart eller helt mulet) har vi lagt till en devil-modell som, genom att jämföra med prognoserna från fem andra vädermodeller (se figur), lär sig när procenten ska dras till helt mulet, så att modellen slutar gissa på en säker mellansiffra.

![Gamla prognoser och SMHI:s observationer tränar modellen, som sedan rättar dagens prognos](presentation/learning-from-forecasts.svg)

![Huvudmodellen säger 70 procent; devilen avgör med hjälp av fem andra vädermodeller om det ska bli 100 procent](presentation/overcast-devil.svg)

## Källor

Väderikoner: Yr/NRK, CC BY 4.0. Prognosdata: Open-Meteo. Observationer: SMHI. Karta: © OpenStreetMap contributors.
