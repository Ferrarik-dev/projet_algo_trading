async function loadDashboardData() {
    try {
        // Cache buster pour forcer GitHub Pages à charger la toute dernière version du JSON
        const response = await fetch(`dashboard_data.json?t=${new Date().getTime()}`);
        if (!response.ok) throw new Error("Fichier de données introuvable");
        const data = await response.json();

        updateHeader(data.timestamp);
        updateKPIs(data.account);
        updateTopPicks(data.top_picks);
        updateFinBERT(data.top_picks);
        
        if (data.account && data.account.history) {
            updateHistory(data.account.history);
        }
        
        if (data.performance) {
            updatePerformance(data.performance);
        }
        
        if (data.full_analysis) {
            updateAnalysisTable(data.full_analysis);
        } else if (data.market_data) {
            updateHeatmap(data.market_data); // Fallback old version
        }

    } catch (error) {
        console.error("Erreur de chargement:", error);
    }
}

function updateHeader(timestamp) {
    document.getElementById('last-update').innerText = timestamp;
}

function updateKPIs(account) {
    if (!account) return;
    const formatCurrency = (num) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(num);
    
    document.getElementById('account-balance').innerText = formatCurrency(account.balance);
    document.getElementById('buying-power').innerText = formatCurrency(account.buying_power);
}

function updateTopPicks(topPicks) {
    const container = document.getElementById('top-picks-grid');
    container.innerHTML = '';

    if (!topPicks || topPicks.length === 0) {
        container.innerHTML = '<p>Aucune action sélectionnée aujourd\'hui.</p>';
        return;
    }

    topPicks.forEach((pick, index) => {
        const card = document.createElement('div');
        card.className = 'pick-card';
        
        let finbertColor = pick.nlp_score > 0.2 ? 'var(--color-up)' : (pick.nlp_score < -0.2 ? 'var(--color-down)' : '#94a3b8');
        let finbertIcon = pick.nlp_score > 0.2 ? '✅' : (pick.nlp_score < -0.2 ? '❌' : '➖');
        
        card.innerHTML = `
            <div class="pick-header">
                <span class="rank">${index + 1}</span>
                <span class="ticker">${pick.ticker}</span>
            </div>
            <div class="prediction">
                <span class="pred-label">Hausse Prévue (5J)</span>
                <span class="pred-value">+${pick.pred_return.toFixed(2)}%</span>
            </div>
            <div style="margin-top: 10px; font-size: 12px; color: ${finbertColor};">
                ${finbertIcon} Score FinBERT : ${pick.nlp_score.toFixed(2)}
            </div>
        `;
        container.appendChild(card);
    });
}

function updateFinBERT(topPicks) {
    if (!topPicks || topPicks.length === 0) return;

    // We analyze the worst score among the top picks to see if there is a veto
    let worstScore = 1;
    let worstAlert = "";
    let allHeadlines = [];

    topPicks.forEach(pick => {
        if (pick.nlp_score < worstScore) {
            worstScore = pick.nlp_score;
            worstAlert = pick.nlp_alert;
        }
        if (pick.headlines && pick.headlines.length > 0) {
            allHeadlines = allHeadlines.concat(pick.headlines.map(h => `[${pick.ticker}] ${h}`));
        }
    });

    const verdictEl = document.getElementById('nlp-verdict');
    verdictEl.className = 'nlp-verdict'; // reset
    verdictEl.innerText = worstAlert || `Score global : ${worstScore.toFixed(2)}`;

    if (worstScore < -0.2) {
        verdictEl.classList.add('verdict-veto');
    } else if (worstScore > 0.2) {
        verdictEl.classList.add('verdict-ok');
    } else {
        verdictEl.classList.add('verdict-neutral');
    }

    // Update headlines
    const headlinesList = document.getElementById('nlp-headlines');
    headlinesList.innerHTML = '';
    allHeadlines.slice(0, 10).forEach(h => {
        const li = document.createElement('li');
        li.innerText = h;
        headlinesList.appendChild(li);
    });
}

function updateHeatmap(marketData) {
    const container = document.getElementById('heatmap-grid');
    container.innerHTML = '';

    if (!marketData) return;

    // Convert object to array and sort by performance
    const assets = Object.keys(marketData).map(ticker => ({
        ticker,
        price: marketData[ticker].price,
        change: marketData[ticker].change_pct
    })).sort((a, b) => b.change - a.change);

    assets.forEach(asset => {
        const box = document.createElement('div');
        box.className = 'heat-box';
        
        // Determine color class
        if (asset.change > 2) box.classList.add('heat-up-high');
        else if (asset.change > 0) box.classList.add('heat-up-low');
        else if (asset.change < -2) box.classList.add('heat-down-high');
        else if (asset.change < 0) box.classList.add('heat-down-low');
        else box.classList.add('heat-neutral');

        box.innerHTML = `
            <span class="heat-ticker">${asset.ticker}</span>
            <span class="heat-value">${asset.change > 0 ? '+' : ''}${asset.change.toFixed(2)}%</span>
        `;
        container.appendChild(box);
    });
}

// Nouvelle fonction: Historique Alpaca
function updateHistory(historyData) {
    const container = document.getElementById('history-list');
    if (!container) return;
    
    container.innerHTML = '';
    
    if (!historyData || historyData.length === 0) {
        container.innerHTML = '<p>Aucune transaction récente.</p>';
        return;
    }
    
    historyData.forEach(trade => {
        const item = document.createElement('div');
        item.className = 'history-item';
        
        const isBuy = trade.side.toLowerCase() === 'buy';
        const sideClass = isBuy ? 'history-buy' : 'history-sell';
        const sideText = isBuy ? 'ACHAT' : 'VENTE';
        
        item.innerHTML = `
            <div class="history-side ${sideClass}">${sideText}</div>
            <div class="history-symbol">${trade.symbol}</div>
            <div class="history-qty">${trade.qty} parts</div>
            <div class="history-price">$${trade.price.toFixed(2)}</div>
            <div class="history-date">${trade.date.split(' ')[0]}</div>
        `;
        container.appendChild(item);
    });
}

// Nouvelle fonction: Tableau Analytique Complet
function updateAnalysisTable(analysisData) {
    const tbody = document.getElementById('analysis-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    if (!analysisData || analysisData.length === 0) return;
    
    analysisData.forEach(asset => {
        const tr = document.createElement('tr');
        
        const isUp = asset.pred_return > 0;
        const colorClass = isUp ? 'col-up' : 'col-down';
        const sign = isUp ? '+' : '';
        
        tr.innerHTML = `
            <td style="font-weight:bold">${asset.ticker}</td>
            <td class="${colorClass}">${sign}${asset.pred_return.toFixed(2)}%</td>
            <td>${asset.rsi.toFixed(1)}</td>
            <td>${asset.macd.toFixed(2)}</td>
            <td>${asset.volatility.toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
    });
}

let perfChartInstance = null;

function updatePerformance(perfData) {
    if (!perfData || !perfData.timestamps || perfData.timestamps.length === 0) return;
    
    const botPerf = document.getElementById('bot-perf');
    const spyPerf = document.getElementById('spy-perf');
    
    const botVal = perfData.bot_return_pct;
    const spyVal = perfData.spy_return_pct;
    
    botPerf.innerText = (botVal >= 0 ? '+' : '') + botVal.toFixed(2) + '%';
    botPerf.style.color = botVal >= 0 ? 'var(--color-up)' : 'var(--color-down)';
    
    spyPerf.innerText = (spyVal >= 0 ? '+' : '') + spyVal.toFixed(2) + '%';
    spyPerf.style.color = spyVal >= 0 ? 'var(--color-up)' : 'var(--color-down)';
    
    const ctx = document.getElementById('perfChart').getContext('2d');
    
    if (perfChartInstance) {
        perfChartInstance.destroy();
    }
    
    perfChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: perfData.timestamps,
            datasets: [
                {
                    label: 'QuantBot IA (Base 100)',
                    data: perfData.bot_equity,
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true,
                    pointRadius: 0,
                    pointHoverRadius: 4
                },
                {
                    label: 'S&P 500 Benchmark',
                    data: perfData.spy_equity,
                    borderColor: '#fbbf24',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.3,
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8' }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)'
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', maxTicksLimit: 8 }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

// Lancer le chargement
loadDashboardData();
// Rafraichir toutes les 60 secondes si le bot tourne en continu
setInterval(loadDashboardData, 60000);
