// app.js - Logique Frontend style QuickBooks

document.addEventListener('DOMContentLoaded', () => {
    fetchDashboardData();
});

async function fetchDashboardData() {
    try {
        const response = await fetch(`dashboard_data.json?t=${new Date().getTime()}`);
        if (!response.ok) throw new Error('Data not found');
        const data = await response.json();
        
        updateAccount(data.account);
        updateTopPicks(data.top_picks);
        updateHistory(data.account.history);
        updatePerformance(data.performance);
        
    } catch (error) {
        console.error("Erreur de chargement des donnees:", error);
    }
}

function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
}

function formatPercent(value) {
    return (value > 0 ? '+' : '') + value.toFixed(2) + '%';
}

function updateAccount(account) {
    if (!account) return;
    
    document.getElementById('account-balance').textContent = formatCurrency(account.balance || 0);
    document.getElementById('buying-power').textContent = formatCurrency(account.buying_power || 0);
}

function updateTopPicks(picks) {
    const container = document.getElementById('top-picks-list');
    const countBadge = document.getElementById('picks-count');
    
    if (!picks || picks.length === 0) {
        container.innerHTML = '<div class="loading-text">Aucun achat en cours</div>';
        countBadge.textContent = '0';
        return;
    }
    
    countBadge.textContent = picks.length;
    container.innerHTML = '';
    
    picks.forEach(pick => {
        const isPos = pick.pred_return > 0;
        const colorClass = isPos ? 'positive' : 'negative';
        
        const html = `
            <div class="list-item">
                <div class="item-left">
                    <span class="item-title">${pick.ticker}</span>
                    <span class="item-subtitle">${pick.nlp_alert.split(':')[0]}</span>
                </div>
                <div class="item-right">
                    <span class="item-value">${formatCurrency(pick.price)}</span>
                    <span class="item-subvalue ${colorClass}">Est. ${formatPercent(pick.pred_return)}</span>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', html);
    });
}

function updateHistory(history) {
    const container = document.getElementById('history-list');
    
    if (!history || history.length === 0) {
        container.innerHTML = '<div class="loading-text">Aucune transaction récente</div>';
        return;
    }
    
    container.innerHTML = '';
    
    // N'afficher que les 5 dernieres
    history.slice(0, 5).forEach(order => {
        const html = `
            <div class="list-item">
                <div class="item-left">
                    <span class="item-title">${order.symbol}</span>
                    <span class="item-subtitle">${order.date}</span>
                </div>
                <div class="item-right">
                    <span class="item-value">${order.side === 'BUY' ? 'Achat' : 'Vente'}</span>
                    <span class="item-subvalue">${order.qty} @ ${formatCurrency(order.price)}</span>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', html);
    });
}

function updatePerformance(perf) {
    if (!perf || !perf.timestamps || perf.timestamps.length === 0) return;

    const botRet = perf.bot_return_pct || 0;
    const botPerfEl = document.getElementById('bot-perf');
    const trendIcon = document.getElementById('trend-icon');
    const trendContainer = document.getElementById('trend-container');
    
    botPerfEl.textContent = formatPercent(botRet);
    
    if (botRet >= 0) {
        trendContainer.className = 'balance-trend positive';
        trendIcon.name = 'arrow-up-outline';
    } else {
        trendContainer.className = 'balance-trend negative';
        trendIcon.name = 'arrow-down-outline';
    }

    const ctx = document.getElementById('perfChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: perf.timestamps,
            datasets: [
                {
                    label: 'Bot IA (Base 100)',
                    data: perf.bot_equity,
                    borderColor: '#2CA01C', // QuickBooks Green
                    backgroundColor: 'rgba(44, 160, 28, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0
                },
                {
                    label: 'S&P 500 (Base 100)',
                    data: perf.spy_equity,
                    borderColor: '#6B6C72', // Neutral Gray
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: false,
                    pointRadius: 0
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
                    position: 'bottom',
                    labels: {
                        color: '#393A3D',
                        font: { family: "'Inter', sans-serif", size: 11 },
                        usePointStyle: true,
                        boxWidth: 6
                    }
                },
                tooltip: {
                    backgroundColor: '#393A3D',
                    titleFont: { family: "'Inter', sans-serif", size: 12 },
                    bodyFont: { family: "'Inter', sans-serif", size: 12 },
                    padding: 10,
                    cornerRadius: 8
                }
            },
            scales: {
                x: {
                    display: false // Hide x axis for cleaner mobile look
                },
                y: {
                    grid: {
                        color: '#E3E5E8',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6B6C72',
                        font: { family: "'Inter', sans-serif", size: 10 }
                    }
                }
            }
        }
    });
}
