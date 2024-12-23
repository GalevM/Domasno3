function showPrediction() {
    // Земаме избори од формата
    const publisher = document.getElementById('issuer').value;
    // const duration = document.getElementById('duration').value;

    // Покажуваме графикон
    const chartContainer = document.getElementById('chart-container');
    chartContainer.style.display = 'block';

    // Генерирање на графикон (тук можеш да го замениш ова со реални податоци)
    const ctx = document.getElementById('predictionChart').getContext('2d');

    // Пример за графикон со фиктивни податоци
    const data = {
        labels: ['Јануари', 'Февруари', 'Март', 'Април', 'Мај', 'Јуни'],
        datasets: [{
            label: 'Цената на акциите',
            data: [100, 150, 120, 180, 160, 200],
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
            fill: false
        }]
    };

    const config = {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            scales: {
                x: {
                    beginAtZero: true
                },
                y: {
                    beginAtZero: true
                }
            }
        }
    };

    // Создавање на графикон
    new Chart(ctx, config);
}