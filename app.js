async function loadDashboardData() {
    try {
        // En prod, ajouter un cache buster: fetch(`dashboard_data.json?t=${new Date().getTime()}`)
        const response = await fetch('dashboard_data.json');
        if (!response.ok) throw new Error("Fichier de données introuvable");
        const data = await response.json();

        updateHeader(data.timestamp);
        updateKPIs(data.account);
        updateTopPicks(data.top_picks);
        updateFinBERT(data.top_picks);
        updateHeatmap(data.market_data);

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
        
        card.innerHTML = `
            <div class="pick-header">
                <span class="rank">${index + 1}</span>
                <span class="ticker">${pick.ticker}</span>
            </div>
            <div class="prediction">
                <span class="pred-label">Hausse Prévue (5J)</span>
                <span class="pred-value">+${pick.pred_return.toFixed(2)}%</span>
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

// Lancer le chargement
loadDashboardData();
// Rafraichir toutes les 60 secondes si le bot tourne en continu
setInterval(loadDashboardData, 60000);
