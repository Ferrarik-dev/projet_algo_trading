let globalData = null;
let perfChartInstance = null;
let spyChartInstance = null;

// Tab Routing
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', function() {
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        this.classList.add('active');
        
        const target = this.getAttribute('data-target');
        document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
        document.getElementById(target).classList.add('active');
    });
});

// Formatters
function formatCurrency(val) {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);
}
function formatPercent(val) {
    const sign = val > 0 ? '+' : '';
    return sign + val.toFixed(2) + '%';
}
function getColorClass(val) {
    if(val > 0) return 'text-green';
    if(val < 0) return 'text-red';
    return 'text-neutral';
}
function getBadgeHTML(val, isCurrency = false) {
    if (val === 0) return `<span class="text-neutral">0</span>`;
    const formatted = isCurrency ? formatCurrency(Math.abs(val)) : Math.abs(val).toFixed(2) + (isCurrency ? '' : '%');
    const sign = val > 0 ? '+' : '-';
    const bgClass = val > 0 ? 'badge-green' : 'badge-red';
    return `<span class="${bgClass}">${sign}${formatted}</span>`;
}

// Fetch JSON
async function loadData() {
    try {
        const response = await fetch('dashboard_data.json?t=' + new Date().getTime());
        globalData = await response.json();
        renderHome();
        renderAssets();
        renderCharts('1M'); // Default
    } catch (e) {
        console.error("Erreur de chargement", e);
    }
}

// Render Home
function renderHome() {
    const data = globalData;
    document.getElementById('total-balance').textContent = formatCurrency(data.account.balance);
    document.getElementById('buying-power').textContent = formatCurrency(data.account.buying_power);
    document.getElementById('last-update').textContent = "Mis à jour: " + data.timestamp;

    // Top Picks
    const picksContainer = document.getElementById('top-picks-list');
    picksContainer.innerHTML = '';
    data.top_picks.forEach(pick => {
        const portfolioPct = pick.portfolio_pct || 0;
        const titleLine = `<span class="item-title">${pick.ticker} <span style="font-size:0.75rem; font-weight:normal; color:#6B6C72">(${portfolioPct.toFixed(1)}% du port.)</span></span>`;
        const nlpStr = pick.nlp_alert.split(':')[0];
        
        picksContainer.innerHTML += `
            <div class="list-item">
                <div class="item-left" style="flex:1">
                    ${titleLine}
                    <span class="item-subtitle">Avis IA: ${nlpStr}</span>
                    <div class="item-row" style="margin-top:12px;">
                        <span>P&L Latent: ${getBadgeHTML(pick.unrealized_pl, true)}</span>
                    </div>
                </div>
                <div class="item-right">
                    <span class="item-value">${formatCurrency(pick.price)}</span>
                    <span class="item-subvalue ${getColorClass(pick.pred_return)}">Prévu: ${formatPercent(pick.pred_return)}</span>
                </div>
            </div>
        `;
    });

    // History
    const historyContainer = document.getElementById('history-list');
    historyContainer.innerHTML = '';
    data.account.history.forEach(order => {
        const sideTxt = order.side.toUpperCase() === 'BUY' ? 'Achat' : 'Vente';
        const color = order.side.toUpperCase() === 'BUY' ? 'text-green' : 'text-red';
        historyContainer.innerHTML += `
            <div class="list-item">
                <div class="item-left">
                    <span class="item-title">${order.symbol}</span>
                    <span class="item-subtitle">${order.date}</span>
                </div>
                <div class="item-right">
                    <span class="item-value ${color}">${sideTxt}</span>
                    <span class="item-subvalue">${order.qty} part(s) @ ${formatCurrency(order.price)}</span>
                </div>
            </div>
        `;
    });
}

// Render Assets (Table)
function renderAssets() {
    const tableBody = document.getElementById('universe-table-body');
    tableBody.innerHTML = '';
    
    if(!globalData.full_analysis) return;
    
    globalData.full_analysis.forEach(asset => {
        tableBody.innerHTML += `
            <tr>
                <td><strong>${asset.ticker}</strong></td>
                <td class="right">${formatCurrency(asset.price)}</td>
                <td class="right">${getBadgeHTML(asset.change_pct, false)}</td>
                <td class="right ${getColorClass(asset.pred_return)}"><strong>${formatPercent(asset.pred_return)}</strong></td>
                <td class="right">${asset.rsi.toFixed(1)}</td>
                <td class="right">${asset.macd.toFixed(2)}</td>
            </tr>
        `;
    });
}

// Filter Chart Data by Range
function filterChartData(perfData, range) {
    let daysToKeep = 0;
    if (range === '5D') daysToKeep = 5;
    if (range === '1M') daysToKeep = 22; // trading days approx
    if (range === '6M') daysToKeep = 130;
    
    if (daysToKeep === 0 || perfData.timestamps.length <= daysToKeep) {
        return perfData;
    }
    
    const startIndex = perfData.timestamps.length - daysToKeep;
    return {
        timestamps: perfData.timestamps.slice(startIndex),
        bot_equity: perfData.bot_equity.slice(startIndex),
        spy_equity: perfData.spy_equity.slice(startIndex)
    };
}

// Render Charts
function renderCharts(range) {
    const rawData = globalData.performance;
    if(!rawData || !rawData.timestamps || rawData.timestamps.length === 0) return;
    
    const slicedData = filterChartData(rawData, range);
    
    const SIMULATED_START_CAPITAL = 10000;
    
    const botStartBase = slicedData.bot_equity[0];
    const spyStartBase = slicedData.spy_equity[0];
    
    const botDollars = slicedData.bot_equity.map(val => (val / botStartBase) * SIMULATED_START_CAPITAL);
    const spyDollars = slicedData.spy_equity.map(val => (val / spyStartBase) * SIMULATED_START_CAPITAL);
    
    // Metrics
    const botEnd = botDollars[botDollars.length - 1];
    const spyEnd = spyDollars[spyDollars.length - 1];
    const botProfit = botEnd - SIMULATED_START_CAPITAL;
    const spyProfit = spyEnd - SIMULATED_START_CAPITAL;
    
    document.getElementById('perf-metrics-div').innerHTML = `
        <div class="perf-metric">
            <div class="perf-metric-title">Gains IA ($10k)</div>
            <div class="perf-metric-val ${getColorClass(botProfit)}">${getBadgeHTML(botProfit, true)}</div>
        </div>
        <div class="perf-metric">
            <div class="perf-metric-title">Gains SPY ($10k)</div>
            <div class="perf-metric-val ${getColorClass(spyProfit)}">${getBadgeHTML(spyProfit, true)}</div>
        </div>
    `;

    // Chart 1: Comparison
    const ctxPerf = document.getElementById('performanceChart').getContext('2d');
    if(perfChartInstance) perfChartInstance.destroy();
    
    perfChartInstance = new Chart(ctxPerf, {
        type: 'line',
        data: {
            labels: slicedData.timestamps,
            datasets: [
                {
                    label: 'Portefeuille IA ($)',
                    data: botDollars,
                    borderColor: '#2CA01C',
                    backgroundColor: 'rgba(44, 160, 28, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.2,
                    pointRadius: 0
                },
                {
                    label: 'S&P 500 ($)',
                    data: spyDollars,
                    borderColor: '#6B6C72',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.2,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 6 } },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) label += ': ';
                            if (context.parsed.y !== null) {
                                label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed.y);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: { display: false },
                y: { display: true, position: 'right', grid: { color: '#F4F5F8' } }
            }
        }
    });

    // Chart 2: SPY Only (Base 100 or actual price)
    const ctxSpy = document.getElementById('spyChart').getContext('2d');
    if(spyChartInstance) spyChartInstance.destroy();
    
    spyChartInstance = new Chart(ctxSpy, {
        type: 'line',
        data: {
            labels: slicedData.timestamps,
            datasets: [
                {
                    label: 'Croissance SPY (Base 100)',
                    data: slicedData.spy_equity,
                    borderColor: '#0077C5',
                    backgroundColor: 'rgba(0, 119, 197, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.2,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { display: false },
                y: { display: true, position: 'right', grid: { color: '#F4F5F8' } }
            }
        }
    });
}

// Dropdown listener
document.getElementById('date-reset-select').addEventListener('change', function() {
    renderCharts(this.value);
});

// Init
window.onload = loadData;
