const supabaseClient = supabase.createClient(
  "https://yxvgwmxlznpxqmmiofuy.supabase.co",
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl4dmd3bXhsem5weHFtbWlvZnV5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE2OTU5NzYsImV4cCI6MjA2NzI3MTk3Nn0.4XZQOkWmI1CLq-FR3KM5sD7ohn0iHdcRqrf5-KFmkho"
);
const currentYear = new Date().getFullYear();
const yearElements = document.querySelectorAll('.current-year');
yearElements.forEach(element => {
    element.textContent = currentYear;
});

const genderDataDiv = document.getElementById('genderData');
const ageDataDiv = document.getElementById('ageData');

let genderChartInstance = null;
let ageChartInstance = null;
let yearlyChartInstance = null;

async function fetchData() {
    try {
    //For current cases
        const { data: currentData, error: currentError } = await supabaseClient
            .from('yearly_record_summary')
            .select('cases')
            .eq('year', currentYear);

        if (currentError) {
            console.error("Error fetching data:", currentError);
            return;
        }
        const totalCases = currentData.reduce((acc, record) => acc + record.cases, 0);   
    // For monthly dengue incidence
        const chunkSize = 1000;
        let allData = [];
        let from = 0;
        let to = chunkSize - 1;
        let keepFetching = true;
        while (keepFetching) {
            const { data: monthlyData, error: monthlyError } = await supabaseClient
                .from('records')
                .select('Month, Cases')
                .eq('Year', currentYear)
                .range(from, to);

            if (monthlyError) {
                console.error('Error fetching data:', monthlyError);
                break;
            }

            allData = allData.concat(monthlyData);
            
            if (monthlyData.length < chunkSize) {
                keepFetching = false;
            } else {
                from += chunkSize;
                to += chunkSize;
            }
        }
        const monthNames = {
            'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5,
            'July': 6, 'August': 7, 'September': 8, 'October': 9, 'November': 10, 'December': 11
        };

        const monthsWithData = Array(12).fill(0);
        allData.forEach(record => {
            const cases = parseFloat(record.Cases);
            const monthName = record.Month;

            if (monthNames[monthName] !== undefined) {
                const monthIndex = monthNames[monthName];
                monthsWithData[monthIndex] += cases;
            } else {
                console.warn(`Invalid month data: ${record.Month}`);
            }
        });

    //For 5-year incidence
        const { data: yearlyData, error: yearlyError } = await supabaseClient
            .from('yearly_record_summary')
            .select('cases, year')
            .in('year', [currentYear, currentYear - 1, currentYear - 2, currentYear - 3, currentYear - 4]);

        if (yearlyError) {
            console.error("Error fetching data:", yearlyError);
            return;
        }
        
        const years = [currentYear - 4, currentYear - 3, currentYear - 2, currentYear - 1, currentYear];
        const totals = [0, 0, 0, 0, 0];
        
        yearlyData.forEach(record => {
            const recordYear = parseInt(record.year, 10);
            const yearIndex = years.indexOf(recordYear);

            if (yearIndex !== -1) {
                totals[yearIndex] += record.cases;
            }
        });

    //For pie charts
        const { data: pieData, error: pieError } = await supabaseClient
            .from('yearly_record_summary')
            .select('male, female, age_0_10, age_11_20, age_21_30, age_31_40, age_41_50, age_51_60, age_61_above')
            .eq('year', currentYear);

        if (pieError) {
            console.error("Data fetch error:", pieError.message);
            return;
        }
                
        let total_male = 0;
        let total_female = 0;
        let total_0_to_10 = 0;
        let total_11_to_20 = 0;
        let total_21_to_30 = 0;
        let total_31_to_40 = 0;
        let total_41_to_50 = 0;
        let total_51_to_60 = 0;
        let total_61_and_above = 0;

        pieData.forEach(row => {
            total_male += parseInt(row.male, 10) || 0;
            total_female += parseInt(row.female, 10) || 0;
            total_0_to_10 += parseInt(row.age_0_10, 10) || 0;
            total_11_to_20 += parseInt(row.age_11_20, 10) || 0;
            total_21_to_30 += parseInt(row.age_21_30, 10) || 0;
            total_31_to_40 += parseInt(row.age_31_40, 10) || 0;
            total_41_to_50 += parseInt(row.age_41_50, 10) || 0;
            total_51_to_60 += parseInt(row.age_51_60, 10) || 0;
            total_61_and_above += parseInt(row.age_61_above, 10) || 0;
        });

        displayDashboard(totalCases, years, totals, monthsWithData, ['January', 'Febuary', 'March', 'April', 'May',
            'June', 'July', 'August', 'September', 'October', 'November', 'December'], total_male, total_female,
            total_0_to_10, total_11_to_20, total_21_to_30, total_31_to_40, total_41_to_50, total_51_to_60, total_61_and_above);
    } catch (err) {
        console.error("Error:", err);
    }
}

function displayDashboard(totalCases, years, totals, monthsWithData, months, total_male, total_female,
            total_0_to_10, total_11_to_20, total_21_to_30, total_31_to_40, total_41_to_50, total_51_to_60, total_61_and_above) {
    document.getElementById('current-cases').textContent = totalCases;
    
    const monthctx = document.getElementById('month-trend').getContext('2d');
    new Chart(monthctx, {
        type: 'bar',
        data: {
            labels: months, 
            datasets: [{
                label: `Dengue Cases in ${currentYear}`, 
                data: monthsWithData, 
                backgroundColor: 'rgba(75, 192, 192, 0.2)', 
                borderColor: 'rgba(75, 192, 192, 1)', 
                borderWidth: 1}]},
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                datalabels: false
            }, 
            scales: {
                x: {
                    display: false
                }, 
                y: {
                    display: false
                }
            }
        }
    });

    const yearctx = document.getElementById('year-trend').getContext('2d');
    new Chart(yearctx, {
        type: "line",
        data: {
            labels: years,
            datasets: [{
                label: 'Total Cases', 
                data: totals, 
                borderColor: '#4BC0C0', 
                fill: false, 
                tension: 0.1}
            ]},
        options: {
            responsiveness: true, 
            plugins: {
                tooltip: {
                    enabled: true
                }, 
                legend: {
                    display: false
                },
                 datalabels: false
            }, 
            scales: {
                x: {
                    display: false
                }, 
                y: {
                    display: false
                }
            }
        }
    });

    const genderctx = document.getElementById('gender-chart').getContext('2d');
    new Chart(genderctx, {
        type: 'pie',
        data: {
        labels: ['Male', 'Female'],
        datasets: [{
            data: [total_male, total_female],
            backgroundColor: ['#36A2EB', '#1e3c72'],
            borderWidth: 0
        }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: {
              display: true,
              position: 'bottom'
            },
            tooltip: {
                enabled: true
            }
          }
        }
    });

    const agectx = document.getElementById('age-chart').getContext('2d');
    new Chart(agectx, {
        type: 'pie',
        data: {
        labels: ['10 and below', '11-20', '21-30', '31-40', '41-50', '51-60', '61 and above'],
        datasets: [{
            data: [total_0_to_10, total_11_to_20, total_21_to_30, total_31_to_40, total_41_to_50, total_51_to_60, total_61_and_above],
            backgroundColor: ['#36A2EB', '#1E3C72', '#FF4500', '#2ECC71', '#FF1493', '#FFD700', '#800080'],
            borderWidth: 0
        }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: {
              display: true,
              position: 'bottom'
            },
            tooltip: {
                enabled: true
            }
          }
        }
    });
    
    document.getElementById('loading').style.display = 'none';
}

async function loadCurrentBarangayMap() {
    try {
        const { data, error } = await supabaseClient
            .from('rate_and_classification')
            .select('Barangay, Year, Month, Week, attack_rate, risk_classification')
            .order('Week', { ascending: false })
            .eq('Year', currentYear);

        if (error) {
            console.error('Error fetching Supabase data:', error);
            return;
        }
        
        const latestRecord = data[0];
        const currentMonth = latestRecord.Month ? latestRecord.Month : 'No Data';
        const currentWeek = latestRecord.Week ? latestRecord.Week : 'N/A';
        const labelElement = document.getElementById('current-period');
        if (labelElement) {
            labelElement.textContent = `Showing data for ${currentMonth} ${currentYear}, Week ${currentWeek}`;
        }
        const latestByBarangay = Object.values(
            data.reduce((acc, record) => {
                if (!acc[record.Barangay] || record.Week > acc[record.Barangay].Week) {
                    acc[record.Barangay] = record;
                }
                return acc;
            }, {})
        );

        const geoResponse = await fetch('assets/SAN_PABLO_MAP.geojson');
        const geojsonData = await geoResponse.json();

        geojsonData.features.forEach(feature => {
            const barangayName = feature.properties.name;
            const barangayData = latestByBarangay.find(
                r => r.Barangay.toLowerCase().trim() === barangayName.toLowerCase().trim()
            );
            feature.properties.risk_classification = barangayData?.risk_classification || "No data yet";
        });

        const map1 = L.map('barangay-map1').setView([14.05, 121.33], 11);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map1);

        L.geoJSON(geojsonData, {
            style: feature => ({
                color: '#1e3c72',
                weight: 0.5,
                opacity: 1,
                fillColor: getColorForRiskClassification(feature.properties.risk_classification),
                fillOpacity: 1
            }),
            onEachFeature: (feature, layer) => {
                layer.bindPopup(`<b>${feature.properties.name}</b><br>Risk: ${feature.properties.risk_classification}`);
                layer.on({
                    mouseover: e => e.target.setStyle({ weight: 3 }),
                    mouseout: e => e.target.setStyle({ weight: 0.5 })
                });
            }
        }).addTo(map1);

    } catch (err) {
        console.error('Error loading static barangay map:', err);
    }
}

// =========Barangay Section=========
async function initializeBarangayUI() {
    try {
        // Year dropdown
        const latestRes = await fetch("/dtect/barangay/latest");
        const latest = await latestRes.json();
        const latestYear = latest.Year;
        const latestWeek = latest.Week;

        const barangayYearSelect = document.getElementById("barangay-year-select");
        const barangayYearsRes = await fetch("/dtect/barangay/years");
        const barangayYears = await barangayYearsRes.json();

        barangayYearSelect.innerHTML = barangayYears
            .map(y => `<option value="${y}" ${y === latestYear ? "selected" : ""}>${y}</option>`)
            .join("");

        // Week dropdown
        const barangayWeekSelect = document.getElementById("barangay-week-select");
        const barangayWeeksRes = await fetch(`/dtect/barangay/weeks/${latestYear}`);
        const barangayWeeks = await barangayWeeksRes.json();

        barangayWeekSelect.innerHTML = barangayWeeks
            .map(w => `<option value="${w}" ${w === latestWeek ? "selected" : ""}>Week ${w}</option>`)
            .join("");

        await loadBarangayData(latestYear, latestWeek);
        await loadBarangayRiskMap(latestYear, latestWeek);

        // Year change - update week dropdown
        barangayYearSelect.addEventListener("change", async () => {
            const selectedYear = parseInt(barangayYearSelect.value);

            const barangayWeeksRes = await fetch(`/dtect/barangay/weeks/${selectedYear}`);
            const newBarangayWeeks = await barangayWeeksRes.json();

            barangayWeekSelect.innerHTML = newBarangayWeeks
                .map(w => `<option value="${w}">Week ${w}</option>`)
                .join("");

            
            const lastWeek = newBarangayWeeks[newBarangayWeeks.length - 1];
            barangayWeekSelect.value = lastWeek;

            await loadBarangayData(selectedYear, lastWeek);
            await loadBarangayRiskMap(selectedYear, lastWeek);
        });

        // Week change - load data
        barangayWeekSelect.addEventListener("change", async () => {
            const selectedYear = parseInt(barangayYearSelect.value);
            const selectedWeek = parseInt(barangayWeekSelect.value);

            await loadBarangayData(selectedYear, selectedWeek);
            await loadBarangayRiskMap(selectedYear, selectedWeek);
        });

    } catch (err) {
        console.error("Initialization Error:", err);
    }
}

async function loadBarangayData(year, week) {
    document.getElementById('barangay-details').innerHTML = 
        `<p>Please enter a barangay name.</p>`;
    try {
        const res = await fetch(`/dtect/barangay/data/${year}/${week}`);
        const data = await res.json();
        if (!data || data.length === 0) return;

        const latest = data[0];

        // Header
        document.getElementById('barangay-heading').textContent =
            `San Pablo City as of ${latest.Month} ${latest.Year}, Week ${latest.Week}`;

        // Table
        const listContainer = document.getElementById('barangay-list-table');

        data.sort((a, b) => {
            if (b.attack_rate !== a.attack_rate) return b.attack_rate - a.attack_rate;
            return a.Barangay.localeCompare(b.Barangay);
        });

        listContainer.innerHTML = `
            ${data.map(b => {
                let color = '';
                if (b.risk_classification === 'Low Risk') color = '#2ECC71';
                else if (b.risk_classification === 'Moderate Risk') color = '#FFD700';
                else if (b.risk_classification === 'High Risk') color = '#FF6347';

                return `
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">${b.Barangay}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">${b.attack_rate.toFixed(2)}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd; color: ${color};">${b.risk_classification}</td>
                </tr>`;
            }).join('')}
        `;

        // Search barangay details
        const searchButton = document.getElementById('search-barangay');
        const searchInput = document.getElementById('barangay-search');
        searchInput.value = '';

        searchButton.onclick = async () => {
            const query = searchInput.value.trim().toLowerCase();

            if (!query) {
                document.getElementById('barangay-details').innerHTML =
                    `<p>Please enter a barangay name.</p>`;
                return;
            }

            const { data: detailData, error: detailError } = await supabaseClient
                .from('records')
                .select('Barangay, Gender, Age_Group, Cases')
                .eq('Year', year)
                .eq('Week', week)
                .ilike('Barangay', `%${query}%`);

            if (detailError) throw detailError;
            if (!detailData || detailData.length === 0) {
                document.getElementById('barangay-details').innerHTML =
                    `<p>No records found.</p>`;
                return;
            }

            const barangay = detailData[0].Barangay.toUpperCase();

            const totalMale = detailData
                .filter(d => d.Gender === 'Male')
                .reduce((a, b) => a + b.Cases, 0);

            const totalFemale = detailData
                .filter(d => d.Gender === 'Female')
                .reduce((a, b) => a + b.Cases, 0);

            const ageGroups = {};
            detailData.forEach(d => {
                ageGroups[d.Age_Group] = (ageGroups[d.Age_Group] || 0) + d.Cases;
            });

            document.getElementById('barangay-details').innerHTML = `
                <h3>${barangay}</h3>
                <div style="text-align: center;">
                    <table style="margin: auto; border-collapse: collapse;">
                        <thead>
                            <tr>
                                <th style="border-bottom: 2px solid #ddd; padding: 6px;">Gender</th>
                                <th style="border-bottom: 2px solid #ddd; padding: 6px;">Total</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td style="padding: 6px;">Male</td>
                                <td style="padding: 6px;">${totalMale}</td>
                            </tr>
                            <tr>
                                <td style="padding: 6px;">Female</td>
                                <td style="padding: 6px;">${totalFemale}</td>
                            </tr>
                        </tbody>
                    </table>

                    <table style="margin: auto; border-collapse: collapse;">
                        <thead>
                            <tr>
                                <th style="border-bottom: 2px solid #ddd; padding: 6px;">Age Group</th>
                                <th style="border-bottom: 2px solid #ddd; padding: 6px;">Total</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${Object.entries(ageGroups).map(([age, total]) => `
                                <tr>
                                    <td style="padding: 6px;">${age}</td>
                                    <td style="padding: 6px;">${total}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        };

    } catch (err) {
        console.error('Error loading barangay data:', err);
    }
}

let barangayRiskMap = null;
async function loadBarangayRiskMap(year, week) {
    try {
        const { data, error } = await supabaseClient
            .from('rate_and_classification')
            .select('Barangay, Year, Month, Week, attack_rate, risk_classification')
            .eq('Year', year)
            .eq('Week', week);

        if (error) {
            console.error('Error fetching Supabase data for barangayRiskMap:', error);
            return;
        }

        const latestByBarangay = Object.values(
            data.reduce((acc, record) => {
                if (!acc[record.Barangay] || record.Week > acc[record.Barangay].Week) {
                    acc[record.Barangay] = record;
                }
                return acc;
            }, {})
        );

        const geoResponse = await fetch('assets/SAN_PABLO_MAP.geojson');
        const geojsonData = await geoResponse.json();

        geojsonData.features.forEach(feature => {
            const barangayName = feature.properties.name;
            const barangayData = latestByBarangay.find(
                r => r.Barangay.toLowerCase().trim() === barangayName.toLowerCase().trim()
            );
            feature.properties.risk_classification = barangayData?.risk_classification || "No data yet";
        });

        if (!barangayRiskMap) {
            barangayRiskMap = L.map('barangay-map2').setView([14.05, 121.32], 11);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap contributors'
            }).addTo(barangayRiskMap);
        }

        // Remove old layers
        barangayRiskMap.eachLayer(layer => {
            if (layer instanceof L.GeoJSON) {
                barangayRiskMap.removeLayer(layer);
            }
        });

        L.geoJSON(geojsonData, {
            style: feature => ({
                color: '#1e3c72',
                weight: 0.5,
                opacity: 1,
                fillColor: getColorForRiskClassification(feature.properties.risk_classification),
                fillOpacity: 1
            }),
            onEachFeature: (feature, layer) => {
                layer.bindPopup(`<b>${feature.properties.name}</b><br>Risk: ${feature.properties.risk_classification}`);
                layer.on({
                    mouseover: e => e.target.setStyle({ weight: 3 }),
                    mouseout: e => e.target.setStyle({ weight: 0.5 })
                });
            }
        }).addTo(barangayRiskMap);

    } catch (err) {
        console.error('Error loading barangayRiskMap:', err);
    }
}
// =========Yearly Section=========
async function updateTotalCases() {
    const response = await fetch("/dtect/yearly/case");
    const data = await response.json();

    const totalCases = Object.values(data).reduce((a, b) => a + b, 0);

    document.getElementById("total-cases-value").textContent = totalCases.toLocaleString();
}


// =========Forecast Section=========
let forecastMap, barangayChart, cityChart;
const loadModal = document.getElementById('load');
async function displayForecast(barangayForecastData, citywideForecastData) {
    try {
        const uniqueDates = [...new Set(barangayForecastData.map(item => item.week_range))];
        const weekSelect = document.getElementById("forecast-week-select");
        let barangayLayer;

        if (forecastMap) {
            forecastMap.remove();
        }

        // Populate week dropdown
        weekSelect.innerHTML = uniqueDates
            .map((d, i) => `<option value="${i + 1}">${d}</option>`)
            .join("");

        forecastMap = L.map("forecast-map").setView([14.05, 121.33], 11);
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: "&copy; OpenStreetMap contributors"
        }).addTo(forecastMap);

        const legend = L.control({ position: "bottomright" });
        legend.onAdd = function () {
            const div = L.DomUtil.create("div", "info legend");
            const riskLevels = ["Low Risk", "Moderate Risk", "High Risk", "No Forecast"];
            const colors = ["#2ECC71", "#FFD700", "#FF6347", "#808080"];
            
            for (let i = 0; i < riskLevels.length; i++) {
                div.innerHTML += `
                    <span style="display: inline-flex; align-items: center; margin-right: 10px;">
                        <i style="
                            background:${colors[i]};
                            border:1px solid #000;
                            width:10px;
                            height:10px;
                            display: inline-block;
                            margin-right:3px;">
                        </i>
                        ${riskLevels[i]}
                    </span>
                `;
            }
            return div;
        };
        legend.addTo(forecastMap);

        // Barangay Chart
        const ctxBarangay = document.getElementById("forecast-chart").getContext("2d");
            
        if (barangayChart) {
            barangayChart.destroy();
        }

        barangayChart = new Chart(ctxBarangay, {
            type: "line",
            data: {
                labels:[],
                datasets: []
            },
            options: {
                responsiveness: true,
                plugins: { 
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const scoreToRisk = { 1: "Low Risk", 2: "Moderate Risk", 3: "High Risk" };
                                const value = context.parsed.y;
                                return `${scoreToRisk[value] || "No Data"}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true,
                        min: 0,
                        max: 4,
                        ticks: {
                            stepSize: 1,
                            callback: value => {
                                const scoreToRisk = { 1: "Low", 2: "Moderate", 3: "High" };
                                return scoreToRisk[value] || "";
                            }
                        }
                    }
                }
            }
        });

        // Citywide Chart
        const ctxCity = document.getElementById("citywide-chart").getContext("2d");
        const riskToScore = { "Low Risk": 1, "Moderate Risk": 2, "High Risk": 3 };
        
        if (cityChart) {
            cityChart.destroy();
        }

        const cityLabels = citywideForecastData.map(d => d.week_range);
        const cityRiskScores = citywideForecastData.map(d => riskToScore[d.predicted_risk] || 0);

        cityChart = new Chart(ctxCity, {
            type: "line",
            data: {
                labels: cityLabels,
                datasets: [{
                    data: cityRiskScores,
                    borderColor: "#1e3c72",
                    fill: false,
                    pointBackgroundColor: citywideForecastData.map(d =>
                        getColorForRiskClassification(d.predicted_risk)
                    )
                }]
            },
            options: {
                responsiveness: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const scoreToRisk = { 1: "Low Risk", 2: "Moderate Risk", 3: "High Risk" };
                                const value = context.parsed.y;
                                return `${scoreToRisk[value] || "No Data"}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true,
                        min: 0,
                        max: 4,
                        ticks: {
                            stepSize: 1,
                            callback: v => ({ 1: "Low", 2: "Moderate", 3: "High" }[v] || "")
                        }
                    }
                }
            }
        });

        const geoResponse = await fetch("assets/SAN_PABLO_MAP.geojson");
        const geojsonData = await geoResponse.json();

        async function updateForecastMap(weekNumber) {
            const selectedDate = uniqueDates[weekNumber - 1];
            const weekData = barangayForecastData.filter(item => item.week_range === selectedDate);

            geojsonData.features.forEach(feature => {
                const barangayName = feature.properties.name;
                const barangayData = weekData.find(
                    rec => rec.Barangay.toLowerCase().trim() === barangayName.toLowerCase().trim()
                );
                feature.properties.predicted_risk = barangayData ? barangayData.predicted_risk : "No Forecast";
            });

            if (barangayLayer) forecastMap.removeLayer(barangayLayer);

            barangayLayer = L.geoJSON(geojsonData, {
                style: feature => ({
                    color: "#1e3c72",
                    weight: 0.5,
                    opacity: 1,
                    fillColor: getColorForRiskClassification(feature.properties.predicted_risk),
                    fillOpacity: 1
                }),
                onEachFeature: (feature, layer) => {
                    const name = feature.properties.name;
                    const risk = feature.properties.predicted_risk;
                    layer.bindPopup(`<b>${name}</b><br>Risk: ${risk}`);
                    layer.on({
                        mouseover: e => e.target.setStyle({ weight: 3 }),
                        mouseout: e => e.target.setStyle({ weight: 0.5 }),
                        click: () => updateChart(name)
                    });
                }
            }).addTo(forecastMap);
        }

        // Barangay Table
        function updateForecastTable(weekNumber) {
            const selectedDate = uniqueDates[weekNumber - 1];
            const weekData = barangayForecastData.filter(item => item.week_range === selectedDate);
            
            const riskOrder = { "High Risk": 1, "Moderate Risk": 2, "Low Risk": 3 };
            const sortedWeekData = weekData.sort( (a, b) => riskOrder[a.predicted_risk] - riskOrder[b.predicted_risk] );
             
            const tableBody = document.getElementById("forecast-list");

            tableBody.innerHTML = `
            <table style="width: 100%; border-collapse: collapse; text-align: left;">
            <thead><th>Barangay</td><th>Predicted Risk</td></thead>
                </tbody>
                    ${sortedWeekData.map(d => {
                        const color = getColorForRiskClassification(d.predicted_risk);
                        return `
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">${d.Barangay}</td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd; color: ${color};">${d.predicted_risk}</td>
                        </tr>
                    `;
                    }).join("")}
                </tbody>
            </table>
            `;
        }
        
        // Barangay Chart - Updating
        function updateChart(barangayName) {
            const barangayData = barangayForecastData.filter(item => item.Barangay === barangayName);
            const labels = barangayData.map(d => d.week_range);
            
            const riskToScore = { "Low Risk": 1, "Moderate Risk": 2, "High Risk": 3 };
            const riskScores = barangayData.map(d => riskToScore[d.predicted_risk] || 0);

            document.getElementById("forecast-barangay").textContent = "Risk Forecast in " + barangayName;

            barangayChart.data.labels = labels;
            barangayChart.data.datasets = [{
                data: riskScores,
                borderColor: "#1e3c72",
                fill: false,
                pointBackgroundColor: barangayData.map(d => getColorForRiskClassification(d.predicted_risk))
            }];
            barangayChart.update();
        }

        weekSelect.addEventListener("change", e => {
            const weekNumber = parseInt(e.target.value);
            updateForecastMap(weekNumber);
            updateForecastTable(weekNumber);
        });

        // Initial load
        document.querySelectorAll(".placeholder-text").forEach(el => el.style.display = "none");
        document.getElementById("week-dropdown").style.display = "block";
        document.querySelector(".barangay-citywide-wrapper").style.display = "flex";
        loadModal.style.display = "none";
        document.getElementById("forecast-description").style.display = "none";
        if (forecastMap) {
            forecastMap.invalidateSize();
            await updateForecastMap(1);
            updateForecastTable(1);
        };

        // Citywide Table
        const cityTableContainer = document.getElementById("citywide-table");
        cityTableContainer.innerHTML = `
        <table style="width:100%; border-collapse:collapse; text-align:left;">
            <thead><th>Week Range</th><th>Predicted Risk</th></thead>
            <tbody>
                ${citywideForecastData.map(row => `
                    <tr>
                        <td style="padding:8px; border-bottom:1px solid #ddd;">${row.week_range}</td>
                        <td style="padding:8px; border-bottom:1px solid #ddd; color:${getColorForRiskClassification(row.predicted_risk)};">
                            ${row.predicted_risk}
                        </td>
                    </tr>
                `).join("")}
            </tbody>
        </table>`;
    } catch (err) {
        console.error("Error displaying forecast:", err);
    }
}

function getColorForRiskClassification(riskClassification) {
    switch (riskClassification.toLowerCase()) {
        case 'low risk':
            return '#2ECC71';
        case 'moderate risk':
            return '#FFD700';
        case 'high risk':
            return '#FF6347';
        case 'no data yet':
        default:
            return '#808080';
    }
}

// =========Editable Contents=========
async function loadContent() {
    const { data, error } = await supabaseClient.from('site_content').select('*');
    if (error) {
        console.error('Error loading content:', error);
        return;
    }
    
    const getValue = (key) => data.find(item => item.key === key)?.value || '';
    document.getElementById('prevention_header_title').textContent = getValue('prevention_header_title');
    document.getElementById('prevention_header_text').textContent = getValue('prevention_header_text');
}

async function loadPreventionContent() {
    const preventionContentContainer = document.getElementById('prevention_content_container');
    const { data, error } = await supabaseClient
        .from('prevention_content')
        .select('*')
        .order('created_at', { ascending: false });

    if (error) {
        console.error('Load error:', error.message);
        return;
    }

    preventionContentContainer.innerHTML = '';

    data.forEach(row => {
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('prevention-item');

        contentDiv.innerHTML = `
            <h3 id="title-${row.id}" class="prevention-content-title"><strong>${row.title}</strong></h3>
            <div id="content-${row.id}" class="prevention-content-text">${row.content}</div>`;

        preventionContentContainer.appendChild(contentDiv);
    });
}

async function loadContactDetails() {
    const container = document.getElementById('contact-details-row');
    if (!container) return;

    const { data, error } = await supabaseClient
        .from('contact_details')
        .select('*')
        .order('created_at', { ascending: true });

    if (error) return console.error(error);

    container.innerHTML = '';

    data.forEach(row => {
        const block = document.createElement('div');
        block.className = 'contact-block';
        block.innerHTML = `
            <h3>${row.office_name}</h3>
            <p>
            ${row.address}<br>
            ${row.phone ? `Phone: <a href="tel:${row.phone}">${row.phone}</a><br>` : ''}
            ${row.email ? `Email: <a href="mailto:${row.email}">${row.email}</a><br>` : ''}
            </p>`;

        container.appendChild(block);
    });
}

// =========Nearby Hospital=========
let map;
function initMap(lat, lon) {
    const sanPabloLat = 14.0685;
    const sanPabloLon = 121.3259;

    map = L.map('hospital_map').setView([sanPabloLat, sanPabloLon], 13);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    enableClickSelection();
}

function getDistance(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(lat1 * Math.PI / 180) *
        Math.cos(lat2 * Math.PI / 180) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

function enableClickSelection() {
    let userMarker = null;
    map.on('click', async e => {
        const { lat, lng } = e.latlng;
        if (userMarker) map.removeLayer(userMarker);

        userMarker = L.marker([lat, lng], { 
            icon: new L.Icon({
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            })
        }).addTo(map).openPopup();

        await getNearbyHospitals(lat, lng);
    });
}

async function getNearbyHospitals(lat, lon) {
    const sanPabloLat = 14.0685;
    const sanPabloLon = 121.3259;

    const res = await fetch(`/dtect/hospitals?lat=${sanPabloLat}&lon=${sanPabloLon}`);
    const data = await res.json();

    const filtered = data.filter(h =>
        h.display_name.toLowerCase().includes("san pablo")
    );

    
    const hospitalsWithDistance = filtered.map(hospital => {
        const hospitalLat = parseFloat(hospital.lat);
        const hospitalLon = parseFloat(hospital.lon);
        const distance = getDistance(lat, lon, hospitalLat, hospitalLon);
        return { ...hospital, distance };
    }).sort((a, b) => a.distance - b.distance);

    const list = document.getElementById('hospital-list');
    list.innerHTML = '';

    hospitalsWithDistance.forEach(hospital => {
        const name = hospital.display_name.split(',')[0].trim();
        const addressParts = hospital.display_name.split(',').slice(1, -3).map(p => p.trim());
        const cleanAddress = addressParts.join(', ');

        const row = document.createElement('tr');
        row.innerHTML = `<td>${name}</td><td>${cleanAddress}</td><td>${hospital.distance.toFixed(2)} km</td>`;
        list.appendChild(row);

        L.marker([hospital.lat, hospital.lon])
            .addTo(map)
            .bindPopup(`<strong>${name}</strong><br>${cleanAddress}<br>${hospital.distance.toFixed(2)} km away`);
    });
}
function showInitialMessage() {
    const list = document.getElementById('hospital-list');
    list.innerHTML = `
        <tr>
            <td colspan="3" style="text-align:center; padding: 20px; color: #555; font-size: medium">
                Click on the map to show nearby hospitals in <strong>San Pablo City</strong>.
            </td>
        </tr>
    `;
}

async function loadVideos() {
  const response = await fetch("/dtect/videos");
  const videos = await response.json();

  const container = document.getElementById('video-slides-row');

  videos.forEach(video => {
    const wrapper = document.createElement('div');
    wrapper.className = 'video-container';
    wrapper.innerHTML = `
        <iframe src="https://www.youtube.com/embed/${video.video_id}?si=eWHSUUX7NLI0x6Pf" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>`;
    container.appendChild(wrapper);
  });
}

document.addEventListener('DOMContentLoaded', () => {
    fetchData();
    loadCurrentBarangayMap();
    initializeBarangayUI();
    updateTotalCases();
    loadContent();
    loadPreventionContent();
    loadContactDetails();
    initMap();
    showInitialMessage();
    loadVideos();

    const sidebar = document.querySelector('.sidebar');
    const hamburgerBtn = document.querySelector('.hamburger-btn');
    const mainContent = document.querySelector('.main-content');
    const navLinks = document.querySelectorAll('.sidebar ul li a');
    const contentSections = document.querySelectorAll('.content-section');
    const overlay = document.getElementById('modal-overlay');
    const infoModal = document.getElementById('info-modal');
    const infoTitle = document.getElementById('info-title');
    const infoText = document.getElementById('info-text');
    const guestBtn = document.getElementById('guest-btn');
    const barangayBtn = document.getElementById('barangay-btn');
    const attackRateBtn = document.getElementById('attack-rate-btn');
    const riskClassificationBtn = document.getElementById('risk-classification-btn');
    const yearBtn = document.getElementById('year-btn');
    const forecastBtn = document.getElementById('forecast-btn');
    const hospitalBtn = document.getElementById('hospital-btn');

    // Sidebar Navigation
    if (sidebar && hamburgerBtn) {
        hamburgerBtn.addEventListener('click', () => {
            sidebar.classList.toggle('active');
        });
    }
    // Highlighting Active Link on Sidebar
    if (mainContent) {
        mainContent.addEventListener('scroll', updateActiveNavLink);
    }
    navLinks.forEach(link => {
        link.addEventListener('click', handleNavLinkClick);
    });
    function handleNavLinkClick(e) {
        e.preventDefault();
        const targetId = e.currentTarget.getAttribute('href');
        const targetElement = document.querySelector(targetId);
        if (targetElement && mainContent) {
            mainContent.scrollTo({ top: targetElement.offsetTop, behavior: 'smooth' });
        }
    }
    function updateActiveNavLink() {
        let currentSectionId = '';
        contentSections.forEach(section => {
            if (section.offsetTop <= mainContent.scrollTop + 150) { 
                currentSectionId = section.getAttribute('id');
            }
        });
        navLinks.forEach(link => {
            link.classList.toggle('active', link.getAttribute('href') === `#${currentSectionId}`);
        });
    }
    
    // Show modal with overlay
    guestBtn.addEventListener('click', (e) => {
        e.preventDefault();
        infoTitle.innerHTML = `About the D-TECT`;
        infoText.innerHTML = `Dengue Tracking and Early Classification Tool <strong>(D-TECT)</strong> is a web-based system designed 
        to help monitor and analyze dengue incidence, and predict dengue risk in <strong>San Pablo City</strong>.<br>
        <br>It was developed to track dengue trends, visualize risk levels, and generate forecasts to support early response. 
        Using machine learning and historical dengue data, D-TECT helps health authorities and the public 
        stay informed about the weekly dengue situation in the city.<br>
        <br>The system offers dashboards, risk maps, detailed data summaries, and forecast tools that work together to 
        support planning, awareness, and community safety.`;
        infoModal.style.display = 'flex';
        overlay.style.display = 'block';
    });
    barangayBtn.addEventListener('click', (e) => {
        e.preventDefault();
        infoTitle.innerHTML = `Barangays Section`;
        infoText.innerHTML = `The Barangays section allows users to explore dengue data for previous weeks.<br>
        <br><strong>How it works:</strong>
        <br>By selecting a year and week from the dropdown, the map automatically updates to show the risk levels during that selected time. 
        <br>This section also displays a table containing the attack rate and risk classification of every barangay for that week.<br>
        <br>Users can also search for a specific barangay to view its detailed data, including gender distribution and age group breakdown. These results change depending on the selected year and week, making it easy to track changes over time.`;
        infoModal.style.display = 'flex';
        overlay.style.display = 'block';
    });
    riskClassificationBtn.addEventListener('click', (e) => {
        e.preventDefault();
        infoTitle.innerHTML = `Risk Classification`;
        infoText.innerHTML = `<p style="color: #FF6347; font-weight: bold;">High Risk</p>
        Epidemic-level activity, well above baseline; requires at least two consecutive weeks of cases.
        <br><em><p style="text-align: center">(> μ + 2σ)</p></em>
        <p style="color: #FFD700; font-weight: bold;">Moderate Risk</p>
        Above-normal activity; upgraded only if cases persist for two consecutive weeks.
        <em><p style="text-align: center">(> μ to ≤ μ + 2σ)</p></em>
        <p style="color: #2ECC71; font-weight: bold;">Low Risk</p>
        Baseline or expected activity; includes<br>isolated cases and non-cases.
        <em><p style="text-align: center">(≤ μ)</p></em>
        <em><p style="font-size: x-small; color: #888888;">Source: <a href="https://apps.who.int/iris/handle/10665/250240" target="_blank">World Health Organization</a></p></em>`;
        infoModal.style.display = 'flex';
        overlay.style.display = 'block';
    });
    attackRateBtn.addEventListener('click', (e) => {
        e.preventDefault();
        infoTitle.innerHTML = `Attack Rate`;
        infoText.innerHTML = `Formula:
        <br><p style="text-align: center;">(Number of new cases / Population at risk) x 100</p>
        <br><em><p style="font-size: x-small; color: #888888;">Source: San Pablo City Health Office</p></em>`;
        infoModal.style.display = 'flex';
        overlay.style.display = 'block';
    });
    yearBtn.addEventListener('click', (e) => {
        e.preventDefault();
        infoTitle.innerHTML = `Summary Section`;
        infoText.innerHTML = `The Summary section gives a broader look at dengue trends across an entire year.<br>
        <br><strong>How it works:</strong>
        <br>By choosing a year from the dropdown, users can view the total recorded dengue cases for that year and even all the recorded cases in the system, 
        a bar graph showing the distribution of cases across barangays, and pie graphs showing the yearly distribution between genders and age groups.<br>
        <br>This section helps users understand long-term patterns and compare how dengue affects different communities.`;
        infoModal.style.display = 'flex';
        overlay.style.display = 'block';
    });
    forecastBtn.addEventListener('click', (e) => {
        e.preventDefault();
        infoTitle.innerHTML = `Forecast Section`;
        infoText.innerHTML = `D-TECT uses historical dengue incidence data together with environmental and climatic variables such as rainfall, 
        temperature, humidity, wind spreed, and wind direction to predict potential risk in the coming weeks.<br>
        <br>By analyzing past patterns and trends, the model estimates the level of dengue risk 
        (classified as Low, Moderate, or High) for both barangay-level and citywide forecasts.<br>
        <br>The system applies time-series and machine learning techniques to identify correlations between climate conditions and dengue transmission, 
        providing insights for early warning and preventive planning.<br>
        <br><strong>How it works:</strong>
        <br>By entering how many weeks they want to forecast, and D-TECT generates two types of forecasts:
        <br><strong>Citywide Forecast</strong> – Shows the predicted dengue risk for the entire city. 
        It includes a table showing week ranges and predicted risk levels, along with a line graph for better visualization.
        <br><strong>Barangay Forecast</strong> – Shows predicted dengue risk for each barangay. 
        The dropdown displays all the weeks selected for forecasting. For each week, 
        users can see a table listing barangays with their predicted risk level and a map with updated risk colors. 
        Clicking on a specific barangay on the map opens a line graph showing the forecast trend for that barangay.<br>
        <br><strong style="color: #d9534f;">Note:</strong> This system provides predictions based on historical data and observed trends,
        and the results should be interpreted as estimates rather than exact outcomes.`;
        infoModal.style.display = 'flex';
        overlay.style.display = 'block';
    });
    hospitalBtn.addEventListener('click', (e) => {
        e.preventDefault();
        infoTitle.innerHTML = `San Pablo City Hospitals`;
        infoText.innerHTML = `<br><strong>How it works:</strong>
        <br>By placing a pin anywhere on the map, D-TECT will automatically display the nearest hospitals in San Pablo City.<br>
        <br>A table also shows each hospital’s name, address, and distance from the pinned location.<br>
        <br>The map helps users visualize how to reach a healthcare facility.`;
        infoModal.style.display = 'flex';
        overlay.style.display = 'block';
    });
    //  Close modal with overlay
    function closeAllModals() {
        if (overlay) overlay.style.display = 'none';
        if (infoModal) infoModal.style.display = 'none';
    }
    // Close modals on overlay click
    if (overlay) {
        overlay.addEventListener('click', closeAllModals);
    }

    (async () => {
        const yearSelect = document.getElementById("year-select");
        const totalCasesEl = document.getElementById("total-cases-value");
        
        let breakdownChart = null;
        let genderChart = null;
        let ageChart = null;

        let yearsData = {};
        let yearlyDetails = {};
        let allBarangayRecords = [];

        const safeNum = v => (typeof v === "number" ? v : Number(v) || 0);
        
        function updateTotalForYear(year, yearsData, breakdownData) {
            let total = 0;

            if (year === "All Years") {
                total = Object.values(breakdownData).reduce((s, v) => s + safeNum(v), 0);
            } else {
                total = Object.values(breakdownData).reduce((s, v) => s + safeNum(v), 0);
            }

            totalCasesEl.textContent = "TOTAL DENGUE CASES RECORDED: " + total.toLocaleString();
        }

        async function fetchYears() {
            try {
                const res = await fetch("/dtect/yearly/city-data");
                if (!res.ok) throw new Error("Failed to fetch yearly data");
                
                return res.json();
            } catch (e) {
                console.error(e);
                return {};
            }
        }

        // Getting data for bar barangay
        async function fetchBreakdown(year) {
            try {
                const url =
                    year === "All Years"
                    ? "/dtect/yearly/barangay-data"
                    : `/dtect/yearly/city-data/${encodeURIComponent(year)}`;
                
                const res = await fetch(url);
                if (!res.ok) throw new Error("Failed to fetch breakdown");
                
                const data = await res.json();

                if (year === "All Years") {
                    if (!allBarangayRecords.length) allBarangayRecords = data;
                    
                    return data.reduce((acc, r) => {
                        const b = r.barangay || r.Barangay || "Unknown";
                        acc[b] = (acc[b] || 0) + safeNum(r.total_cases || r.total_cases_sum);
                        return acc;
                    }, {});
                }
                return data;
            } catch (e) {
                console.error(e);
                return {};
            }
        }

        // Bar radiobutton
        let sortType = "alpha";
        document.querySelectorAll('input[name="sortType"]').forEach(radio => {
            radio.addEventListener("change", () => {
                sortType = radio.value;
                renderBreakdownChart(yearSelect.value);
            });
        });

        // Render bar barangay
        async function renderBreakdownChart(year) {
            const breakdownData = await fetchBreakdown(year);

            let sortedEntries = Object.entries(breakdownData);

            if (sortType === "alpha") {
                sortedEntries.sort((a, b) => a[0].localeCompare(b[0]));
            } else if (sortType === "cases") {
                sortedEntries.sort((a, b) => safeNum(b[1]) - safeNum(a[1]));
            }
            const labels = sortedEntries.map(e => e[0]);
            const data = sortedEntries.map(e => safeNum(e[1]));

            if (breakdownChart) breakdownChart.destroy();

            const ctx = document.getElementById("breakdown-chart").getContext("2d");
            
            breakdownChart = new Chart(ctx, {
                type: "bar",
                data: {
                    labels,
                    datasets: [{
                        label: `Cases (${year})`,
                        data,
                        backgroundColor: "#1e3c72",
                        minBarLength: 3
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: { y: { beginAtZero: true }, x: { display: false }}
                }
            });
            return breakdownData; 
        }

        // Getting data for pie age and gender
        async function fetchYearlyDetails() {
                const res = await fetch("/dtect/yearly/city-data");
                if (!res.ok) throw new Error("Failed to fetch yearly details");
                return await res.json();
        }

        // Render pie gender
        function renderGenderChart(record) {
            const ctx = document.getElementById("genderChart").getContext("2d");

            if (genderChart) genderChart.destroy();
            if (!record) return;

            genderChart = new Chart(ctx, {
                type: "pie",
                data: {
                    labels: ["Male", "Female"],
                    datasets: [{
                        data: [safeNum(record.male), safeNum(record.female)],
                        backgroundColor: ['#36A2EB', '#1e3c72'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: (chart) => {
                                return window.innerWidth <= 650 ? "bottom" : "right";
                            }
                        }
                    }
                }
            });
                        
            window.addEventListener("resize", () => {
                if (genderChart) genderChart.update();
            });
        }

        // Render pie age
        function renderAgeChart(record) {
            const ctx = document.getElementById("ageChart").getContext("2d");

            if (ageChart) ageChart.destroy();
            if (!record) return;

            ageChart = new Chart(ctx, {
                type: "pie",
                data: {
                    labels: [
                        "0–10", "11–20", "21–30",
                        "31–40", "41–50", "51–60", "61+"
                    ],
                    datasets: [{
                        data: [
                            safeNum(record.age_0_10),
                            safeNum(record.age_11_20),
                            safeNum(record.age_21_30),
                            safeNum(record.age_31_40),
                            safeNum(record.age_41_50),
                            safeNum(record.age_51_60),
                            safeNum(record.age_61_above)
                        ],
                        backgroundColor: ['#36A2EB', '#1E3C72', '#FF4500', '#2ECC71', '#FF1493', '#FFD700', '#800080'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: (chart) => {
                                return window.innerWidth <= 650 ? "bottom" : "right";
                            }
                        }
                    }
                }
            });
            window.addEventListener("resize", () => {
                if (ageChart) ageChart.update();
            });
        }

        // All  years - pie gender and age
        function sumAllYearsDetails() {
            const result = {
                male: 0, female: 0,
                age_0_10: 0, age_11_20: 0, age_21_30: 0,
                age_31_40: 0, age_41_50: 0, age_51_60: 0, age_61_above: 0
            };

            Object.values(yearlyDetails).forEach(r => {
                for (const key in result) {
                    result[key] += safeNum(r[key]);
                }
            });

            return result;
        }
        
        // Dropdown option filling
        function buildYearDropdown(years) {
            yearSelect.innerHTML = "";

            const allOption = document.createElement("option");
            allOption.value = "All Years";
            allOption.textContent = "All Years";
            yearSelect.appendChild(allOption);

            years.forEach(y => {
                const opt = document.createElement("option");
                opt.value = y;
                opt.textContent = y;
                yearSelect.appendChild(opt);
            });

            yearSelect.value = "All Years";
        }

        // Initialize charts
        async function init() {
            yearsData = await fetchYears();
            yearlyDetails = await fetchYearlyDetails();

            const yearKeys = Object.keys(yearsData).sort((a, b) => a - b);

            buildYearDropdown(yearKeys);

            const breakdownData = await renderBreakdownChart("All Years");
            updateTotalForYear("All Years", yearsData, breakdownData);

            const summed = sumAllYearsDetails();
            renderGenderChart(summed);
            renderAgeChart(summed);
        }
        
        // Select change
        yearSelect.addEventListener("change", async () => {
            const selected = yearSelect.value;

            const breakdownData = await renderBreakdownChart(selected);
            updateTotalForYear(selected, yearsData, breakdownData);

            if (selected === "All Years") {
                const combined = sumAllYearsDetails();
                renderGenderChart(combined);
                renderAgeChart(combined);
            } else {
                renderGenderChart(yearlyDetails[selected]);
                renderAgeChart(yearlyDetails[selected]);
            }
        });

        await init();
    })();

    document.getElementById("generate-forecast").addEventListener("click", async () => {
        const weeksInput = document.getElementById("weeks-input");
        const weeks = parseInt(weeksInput.value, 10);
        const spinnerText = document.getElementById('spinner-text');
        const loadSpin = document.getElementById('loadSpin');

        loadModal.style.display = "flex";

        if (isNaN(weeks) || weeks < 1) {
            loadSpin.style.display = "none"
            spinnerText.innerHTML = "Please enter a valid number of weeks (1 or more).";
            setTimeout(() => {
                loadSpin.style.display = "block"
                loadModal.style.display = "none";
            }, 3000);
            return;
        }

        spinnerText.innerHTML = `Generating ${weeks}-Week Forecast`;
        
        const barangayResponse = await fetch(`/forecast?mode=barangay&weeks=${weeks}`);
        const barangayResult = await barangayResponse.json();

        const cityResponse = await fetch(`/forecast?mode=citywide&weeks=${weeks}`);
        const cityResult = await cityResponse.json();

        if (barangayResult.success && barangayResult.data) {
            displayForecast(barangayResult.data, cityResult.data && cityResult.success && cityResult.data);
        } else {
            console.error("Forecast failed:", barangayResult.error, cityResult.error);
        }
    });

}); 
