const statusModal = document.getElementById("status-modal");
const statusTitle = document.getElementById('status-title');
const statusContent = document.getElementById('status-content');

async function verifyAuthToken() {
    try {
        const res = await fetch('/dtect/verify-auth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include'
        });
        const data = await res.json();
        if (!data.success) {
            window.location.href = "/";
        }
    } catch (err) {
        console.error("Auth check failed:", err);
        window.location.href = "/";
    }
} 

function showStatus(title, message, options = {}) {
    const {
        duration = 3000,
        showButton = false,
        buttonText = "Okay",
        callback = null,
        cancelButtonText = null,
        cancelCallback = null
    } = options;

    if (!statusModal || !statusTitle || !statusContent) return;

    statusTitle.innerHTML = title;
    statusContent.innerHTML = message;

    const confirmBtn = document.getElementById("status-ok-btn");
    const cancelBtn = document.getElementById("status-cancel-btn");

    confirmBtn.style.display = "none";
    cancelBtn.style.display = "none";

    const newConfirm = confirmBtn.cloneNode(true);
    const newCancel = cancelBtn.cloneNode(true);
    confirmBtn.parentNode.replaceChild(newConfirm, confirmBtn);
    cancelBtn.parentNode.replaceChild(newCancel, cancelBtn);

    const okButton = document.getElementById("status-ok-btn");
    const cancelButton = document.getElementById("status-cancel-btn");

    if (showButton) {
        okButton.textContent = buttonText;
        okButton.style.display = "inline-block";
        okButton.addEventListener("click", () => {
            statusModal.style.display = "none";
            statusTitle.innerHTML = "";
            statusContent.innerHTML = "";
            if (typeof callback === "function") callback();
        });

        if (cancelButtonText) {
            cancelButton.textContent = cancelButtonText;
            cancelButton.style.display = "inline-block";
            cancelButton.addEventListener("click", () => {
                statusModal.style.display = "none";
                statusTitle.innerHTML = "";
                statusContent.innerHTML = "";
                if (typeof cancelCallback === "function") cancelCallback();
            });
        }
    }

    statusModal.style.display = "flex";

    if (!showButton) {
        setTimeout(() => {
            statusModal.style.display = "none";
            statusTitle.innerHTML = "";
            statusContent.innerHTML = "";
            if (typeof callback === "function") callback();
        }, duration);
    }
}

verifyAuthToken();

document.addEventListener('DOMContentLoaded', () => { 
    const spinnerText = document.getElementById("spinner-text");
    const spinner = document.getElementById("spinner");
    const loadingModal = document.getElementById('loading');
    const infoModal = document.getElementById('info-modal');
    const infoBtn = document.getElementById('info-btn');
    const resetBtn = document.getElementById('reset-btn');
    const forgotForm = document.getElementById('forgot-form');
    const forgotPassword = document.getElementById('forgot-password');
    const uploadModal = document.getElementById("upload-modal");
    const uploadBtn = document.getElementById("upload-btn");
    const showDataModal = document.getElementById("showData-Modal");
    const showDataBtn = document.getElementById("showData-btn");
    const barangaySelect = document.getElementById("barangay");

    // Open Modals
    if (infoBtn && infoModal) {
        infoBtn.addEventListener('click', (e) => {
            e.preventDefault();
            infoModal.style.display = 'flex';
        });
    }

    if (resetBtn && forgotPassword) {
        resetBtn.addEventListener('click', (e) => {
            e.preventDefault();
            forgotPassword.style.display = 'flex';
        });
    }

    if (uploadBtn && uploadModal) {
        uploadBtn.addEventListener('click', (e) => {
            e.preventDefault();
            uploadModal.style.display = 'flex'
        })
    }
    
    if (showDataBtn && showDataModal) {
        showDataBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const barangayContainer = document.getElementById("filterBarangay");
            const weekContainer = document.getElementById("filterWeek");
            const yearContainer = document.getElementById("filterYear");

            barangayContainer.innerHTML = "";
            yearContainer.innerHTML = "";
            weekContainer.innerHTML = "";

                    
            const allBarangayOption = document.createElement("div");
            allBarangayOption.innerHTML = `
                <label><input type="checkbox" value="All"> All</label>
            `;
            barangayContainer.appendChild(allBarangayOption);

            allowedBarangays.forEach((b) => {
                const div = document.createElement("div");
                div.innerHTML = `<label><input type="checkbox" value="${b}"> ${b}</label>`;
                barangayContainer.appendChild(div);
            });

            const currentYear = new Date().getFullYear();
            const allYearOption = document.createElement("div");
            allYearOption.innerHTML = `
                <label><input type="checkbox" value="All"> All</label>
            `;
            yearContainer.appendChild(allYearOption);

            for (let y = 2021; y <= currentYear; y++) {
                const div = document.createElement("div");
                div.innerHTML = `<label><input type="checkbox" value="${y}"> ${y}</label>`;
                yearContainer.appendChild(div);
            }

            const allWeekOption = document.createElement("div");
            allWeekOption.innerHTML = `
                <label><input type="checkbox" value="All"> All</label>
            `;
            weekContainer.appendChild(allWeekOption);

            for (let w = 1; w <= 52; w++) {
                const div = document.createElement("div");
                div.innerHTML = `<label><input type="checkbox" value="${w}"> Week ${w}</label>`;
                weekContainer.appendChild(div);
            }
            ["filterBarangay", "filterYear", "filterWeek"].forEach(id => {
                const container = document.getElementById(id);
                const checkboxes = container.querySelectorAll('input[type="checkbox"]');
                const allCheckbox = checkboxes[0];

                allCheckbox.addEventListener('change', () => {
                    checkboxes.forEach(cb => cb.checked = allCheckbox.checked);
                });

                checkboxes.forEach(cb => {
                    if (cb.value !== "All") {
                        cb.addEventListener('change', () => {
                            if (!cb.checked) {
                                allCheckbox.checked = false;
                            } else {
                                const allChecked = Array.from(checkboxes).slice(1).every(c => c.checked);
                                allCheckbox.checked = allChecked;
                            }
                        });
                    }
                });
            });

            showDataModal.style.display = 'flex'
        })
    }
    // To Close Modals (click outside)
    window.addEventListener('click', (e) => {
        if (e.target === infoModal || e.target === forgotPassword || e.target === uploadModal || e.target === showDataModal) {
            if (infoModal) infoModal.style.display = 'none';
            if (forgotPassword) forgotPassword.style.display = 'none';
            if (forgotForm) forgotForm.reset();
            if (uploadModal) uploadModal.style.display = 'none';
            if (showDataModal) showDataModal.style.display = 'none';
        }
    });
    
    // Reset password
    if (forgotForm && forgotPassword) {
        forgotForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const email = forgotPassword.querySelector('#reset-email').value;
            
            showStatus("Reset Password", "You need to log out to reset your password. Proceed?", {
                showButton: true,
                buttonText: "Proceed",
                cancelButtonText: "Cancel",
                callback: async () => {
                    try {
                        const res = await fetch('/api/reset-password', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ email })
                        });

                        const data = await res.json();

                        if (!res.ok) throw new Error(data.error || 'Failed to reset password');

                        showStatus("Reset Password", data.message, {
                            showButton: false,
                            callback: () => {
                                forgotPassword.style.display = 'none';
                                forgotForm.reset();
                                window.location.href = '/logout';
                            }
                        });

                    } catch (err) {
                        showStatus("Error", err.message, { showButton: true });
                    }
                }
            });
        });
    }

    // Log out
    if (logout) {
        logout.addEventListener('click', () => {
            showStatus("Log Out", "Are you sure you want to log out?", {
                showButton: true,
                buttonText: "Log Out",
                cancelButtonText: "Cancel",
                callback: () => {
                    window.location.href = '/logout';
                }
            });
        });
    }

    const allowedBarangays = [
        "Atisan","Bautista","Concepcion","Del Remedio","Dolores",
        "I-A","I-B","II-A","II-A (Bagong Bayan)","II-B","II-C","II-D","II-E","II-F",
        "III-A","III-B","III-C","III-D","III-E","III-F",
        "IV-A","IV-B","IV-C",
        "San Antonio 1","San Antonio 2","San Bartolome","San Buenaventura","San Crispin",
        "San Cristobal","San Diego","San Francisco","San Gabriel","San Gregorio",
        "San Ignacio","San Isidro","San Joaquin","San Jose","San Juan","San Lorenzo",
        "San Lucas 1","San Lucas 2","San Marcos","San Mateo","San Miguel","San Nicolas",
        "San Pedro","San Rafael","San Roque","San Vicente","Santa Ana","Santa Catalina",
        "Santa Cruz","Santa Elena","Santa Felomina","Santa Isabel","Santa Maria",
        "Santa Maria Magdalena","Santa Monica","Santa Veronica","Santiago I","Santiago II",
        "Santisimo Rosario","Santo Angel","Santo Cristo","Santo Niño","Soledad",
        "V-A","V-B","V-C","V-D","VI-A","VI-B","VI-C (Bagong Pook)","VI-D","VI-E",
        "VII-A","VII-B","VII-C","VII-D","VII-E"
    ];

    allowedBarangays.forEach(b => {
        const opt = document.createElement("option");
        opt.value = b;
        opt.textContent = b;
        barangaySelect.appendChild(opt);
    });

    // Upload data
    document.getElementById("save-btn").addEventListener("click", async () => {
        const barangay = document.getElementById("barangay").value;
        const year = document.getElementById("year").value;
        const week = document.getElementById("week").value;
        const population = document.getElementById("population").value;
        const fileInput = document.getElementById("csvFile");

        if (!barangay || !year || !week) {
            return showStatus("Upload Data Failed", "Please fill out all details before uploading.", { showButton: true });
        }
        if (!fileInput.files.length) {
            return showStatus("Upload Data Failed", "Please select a CSV file before proceeding.", { showButton: true });
        }
        if (!population || isNaN(population) || parseInt(population) <= 0) {
            return showStatus("Upload Data Failed", "Population must be a valid positive integer.", { showButton: true });
        }

        
        showStatus("Upload Data", "Are you sure you want to upload this file?", {
            showButton: true,
            buttonText: "Proceed",
            cancelButtonText: "Cancel",
            callback: async () => {
                const text = await fileInput.files[0].text();

                loadingModal.style.display = 'flex';
                spinner.style.display = "block";
                spinnerText.textContent = "Uploading data";

                try {
                    const res = await fetch("/api/upload", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ barangay, year, week, population, csv: text })
                    });
                    const result = await res.json();

                    if (!res.ok || (result && result.success === false)) {
                        throw new Error(result.message || "Upload failed.");
                    }

                    showStatus(result.title || "Upload Failed", result.message, {
                        showButton: true,
                        buttonText: "Okay",
                        callback: () => {
                            uploadModal.style.display = "none";
                        }
                    });
                } catch (err) {
                    console.error(err);
                    showStatus("Upload Failed", err.message || "An unexpected error occurred during upload.", {
                        showButton: true
                    });
                } finally {
                    spinner.style.display = "none";
                    spinnerText.textContent = "";
                    setTimeout(() => {
                        loadingModal.style.display = "none";
                    }, 300);
                }
            }
        });
    });

    let lastFetchedData = [];

    // Get checked values
    const getCheckedValues = (containerId) => {
        const checked = Array.from(document.querySelectorAll(`#${containerId} input[type="checkbox"]:checked`))
            .map(cb => cb.value);
        return checked.includes("All") || checked.length === 0 ? ["All"] : checked;
    };

    // Search data
    document.getElementById("searchDataBtn").addEventListener("click", async () => {
        loadingModal.style.display = 'flex';
        spinner.style.display = "block";
        spinnerText.textContent = "Fetching data";

        const filtersId = ["filterBarangay", "filterYear", "filterWeek"];
        for (const id of filtersId) {
            const container = document.getElementById(id);
            const checkboxes = container.querySelectorAll('input[type="checkbox"]');
            if (!Array.from(checkboxes).some(cb => cb.checked)) {
                showStatus("Search Data", `Please select at least one option in ${id.replace("filter", "")} before searching.`, {
                    showButton: true
                });
                setTimeout(() => {
                    loadingModal.style.display = "none";
                }, 300);
                return;
            }
        }

        const barangays = getCheckedValues("filterBarangay");
        const years = getCheckedValues("filterYear");
        const weeks = getCheckedValues("filterWeek");

        const dataTableContainer = document.getElementById("dataTable-container");
        const tbody = document.getElementById("dataTableBody");

        dataTableContainer.style.display = "block";
        tbody.innerHTML = `
            <tr>
                <td colspan="7" style="text-align:center; padding:20px;">
                    <div class="search-spinner"></div>
                    <p>Loading data...</p>
                </td>
            </tr>
        `;

        try {
            const response = await fetch('/api/searchRecords', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ barangays, years, weeks })
            });

            const result = await response.json();

            if (response.ok) {
                lastFetchedData = result.data || [];
                tbody.innerHTML = "";

                lastFetchedData.forEach(row => {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td>${row.Barangay}</td>
                        <td>${row.Year}</td>
                        <td>${row.Month}</td>
                        <td>${row.Week}</td>
                        <td>${row.Gender}</td>
                        <td>${row.Age_Group}</td>
                        <td>${row.Cases}</td>
                    `;
                    tbody.appendChild(tr);
                });

                
                document.getElementById("showData-Modal-content").scrollTop = dataTableContainer.offsetTop;
                document.getElementById("deleteDataBtn").disabled = lastFetchedData.length === 0;
                document.getElementById("downloadCsvBtn").disabled = lastFetchedData.length === 0;
            } else {
                throw new Error(result.error || "Unknown error");
            }
        } catch (err) {
            console.error(err);
            showStatus("Search Data", `Error fetching data: ${err.message}`, {
                showButton: true
            });
        } finally {
            loadingModal.style.display = "none";
            spinnerText.textContent = "";
            spinner.style.display = "none";
        }
    });

    // Download data
    document.getElementById("downloadCsvBtn").addEventListener("click", () => {
        if (!lastFetchedData.length) {
            showStatus("Download CSV", "No data available to download. Please search first.", {
                showButton: true
            });
            return;
        }

        const timestamp = new Date().toISOString().split("T")[0];
        const safeName = `records_${timestamp}.csv`;

        const headers = Object.keys(lastFetchedData[0]);
        const csvRows = [headers.join(",")];

        for (const row of lastFetchedData) {
            const values = headers.map(h => `"${String(row[h] ?? "").replace(/"/g, '""')}"`);
            csvRows.push(values.join(","));
        }

        const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = safeName;
        a.click();
        URL.revokeObjectURL(url);
    });

    // Delete data
    document.getElementById("deleteDataBtn").addEventListener("click", async () => {
        showStatus("Delete Records", "Are you sure you want to delete these records? This cannot be undone.", {
            showButton: true,
            buttonText: "Delete",
            cancelButtonText: "Cancel",
            callback: async () => {
                loadingModal.style.display = "flex";
                spinnerText.textContent = "Deleting data";
                spinner.style.display = "block";

                const barangays = Array.from(document.querySelectorAll(`#filterBarangay input:checked`)).map(cb => cb.value);
                const years = Array.from(document.querySelectorAll(`#filterYear input:checked`)).map(cb => cb.value);
                const weeks = Array.from(document.querySelectorAll(`#filterWeek input:checked`)).map(cb => cb.value);

                try {
                    const response = await fetch('/api/deleteRecords', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ barangays, years, weeks })
                    });

                    const result = await response.json();
                    if (!response.ok) throw new Error(result.error || "Unknown error");

                    showStatus("Delete Records", "Records deleted successfully.", { showButton: true });

                    document.getElementById("dataTableBody").innerHTML = "";
                    document.getElementById("deleteDataBtn").disabled = true;
                    document.getElementById("downloadCsvBtn").disabled = true;
                    lastFetchedData = [];
                } catch (err) {
                    console.error(err);
                    showStatus("Delete Records", "Error deleting records: " + err.message, { showButton: true });
                } finally {
                    loadingModal.style.display = "none";
                    spinnerText.textContent = "";
                    spinner.style.display = "none";
                }
            }
        });
    });

    const downloadModal = document.getElementById("downloadModal");
    const openBtn = document.getElementById("open-download-modal");
    const barangayList = document.getElementById("barangay-wide-list");
    const barangayTitle = document.getElementById("barangay-wide-list-title");
    const modeSelect = document.getElementById("forecast-mode-select");
    const weeksInput = document.getElementById("forecast-weeks-input");

    // Open modal
    openBtn.onclick = async () => {
        downloadModal.style.display = "block";
        await loadBarangays();
    };

    window.onclick = (e) => {
        if (e.target === downloadModal) downloadModal.style.display = "none";
    };

    // Fetch barangay list from server
    async function loadBarangays() {
        barangayList.innerHTML = `
            <label><input type="checkbox" id="select-all"> Select All</label>
            ${allowedBarangays
                .map(
                    (b) =>
                        `<label><input type="checkbox" class="barangay-checkbox" value="${b}"> ${b}</label>`
                )
                .join("")}
        `;

        const selectAll = document.getElementById("select-all");
        selectAll.addEventListener("change", () => {
            document.querySelectorAll(".barangay-checkbox")
                .forEach((cb) => (cb.checked = selectAll.checked));
        });
    }

    modeSelect.addEventListener("change", () => {
        if (modeSelect.value === "citywide") {
            barangayList.style.display = "none";
            barangayTitle.style.display = "none";
        } else {
            barangayList.style.display = "block";
            barangayTitle.style.display = "block";
        }
    });

    // Download table
    document.getElementById("download-table").addEventListener("click", async () => {
        spinnerText.innerText = "Preparing table";
        loadingModal.style.display = "flex";
        spinner.style.display = "block";
        
        const mode = modeSelect.value;
        const weeks = parseInt(weeksInput.value, 10);
        const selected = [...document.querySelectorAll(".barangay-checkbox:checked")].map(
            (cb) => cb.value
        );

        if (isNaN(weeks) || weeks < 1) {
            showStatus("Download Failed", "Please enter a valid number of weeks (1 or more).", {
                showButton: true,
                buttonText: "Okay"
            });
            loadingModal.style.display = "none";
            spinner.style.display = "none";
            return;
        }

        if (mode === "barangay" && selected.length === 0) {
            showStatus("Download Failed", "Please select at least one barangay.", {
                showButton: true,
                buttonText: "Okay"
            });
            loadingModal.style.display = "none";
            spinner.style.display = "none";
            return;
        }
        
        try {
            let data = [];
            
            const res = await fetch(`/forecast?mode=${mode}&weeks=${weeks}`);
            const result = await res.json();

            if (!result.success) throw new Error("Forecast data unavailable.");

            data = result.data;

            if (mode === "barangay") {
                data = data.filter(d => selected.includes(d.Barangay));
            }

            const csv = [
                "Area,Week Range,Predicted Risk",
                ...data.map(
                    (row) => `${row.Barangay || "San Pablo City"},"${row.week_range}","${row.predicted_risk}"`
                ),
            ].join("\n");

            downloadCSV(csv, `${mode}_${weeks}-week_forecast_.csv`);
            showStatus("Download Complete", "CSV file has been downloaded successfully.", { showButton: true });
        } catch (err) {
            showStatus("Download Failed", err.message, { showButton: true });
        } finally {
            loadingModal.style.display = "none";
            spinner.style.display = "none";
        }
    });

    function downloadCSV(csv, filename) {
        const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(url);
    }

    // Download chart
    document.getElementById("download-chart").addEventListener("click", async () => {
        spinnerText.innerText = "Preparing chart";
        loadingModal.style.display = "flex";
        spinner.style.display = "block";

        const mode = modeSelect.value;
        const weeks = parseInt(weeksInput.value, 10);
        const selected = [...document.querySelectorAll(".barangay-checkbox:checked")].map(
            (cb) => cb.value
        );

        if (isNaN(weeks) || weeks < 1) {
            showStatus("Download Failed", "Please enter a valid number of weeks (1 or more).", {
                showButton: true,
                buttonText: "Okay"
            });
            loadingModal.style.display = "none";
            spinner.style.display = "none";
            return;
        }

        if (mode === "barangay" && selected.length === 0) {
            showStatus("Download Failed", "Please select at least one barangay.", {
                showButton: true,
                buttonText: "Okay"
            });
            loadingModal.style.display = "none";
            spinner.style.display = "none";
            return;
        }
        
        try {
            const riskToScore = { "Low Risk": 1, "Moderate Risk": 2, "High Risk": 3 };

            const res = await fetch(`/forecast?mode=${mode}&weeks=${weeks}`);
            const result = await res.json();

            if (!result.success) throw new Error("Forecast data unavailable.");
            const data = result.data;

            if (mode === "barangay") {
                for (const barangay of selected) {
                    const barangayData = data.filter((d) => d.Barangay === barangay);
                    const labels = barangayData.map((d) => d.week_range);
                    const scores = barangayData.map((d) => riskToScore[d.predicted_risk] || 0);

                    await downloadChartImageFromData({
                        labels,
                        dataPoints: scores,
                        label: `${barangay} ${weeks}-Week Forecast`,
                    }, `${barangay}_${weeks}-week_forecast.png`);
                }
            } else { // Citywide
                const labels = data.map((d) => d.week_range);
                const scores = data.map((d) => riskToScore[d.predicted_risk] || 0);

                await downloadChartImageFromData({
                    labels,
                    dataPoints: scores,
                    label: `San Pablo City ${weeks}-Week Forecast`,
                }, `San_Pablo_City_${weeks}-week_forecast.png`);
            }
            showStatus("Download Complete", "Chart(s) downloaded successfully.", { showButton: true });
        } catch (err) {
            showStatus("Download Failed", err.message, { showButton: true });
        } finally {
            loadingModal.style.display = "none";
            spinner.style.display = "none";
        }
    });

    // Chart download helper
    async function downloadChartImageFromData({ labels, dataPoints, label }, filename) {
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = 800;
        tempCanvas.height = 400;
        document.body.appendChild(tempCanvas);

        const ctx = tempCanvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

        const chart = new Chart(ctx, {
            type: "line",
            data: {
                labels,
                datasets: [{
                    label,
                    data: dataPoints,
                    borderColor: "#1e3c72",
                    fill: false,
                    pointBackgroundColor: dataPoints.map(score => {
                        if (score === 1) return "#2ECC71";
                        if (score === 2) return "#FFD700";
                        if (score === 3) return "#FF6347";
                        return "#808080";
                    })
                }],
            },
            options: {
                responsiveness: false,
                animation: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        min: 0,
                        max: 4,
                        ticks: {
                            stepSize: 1,
                            callback: (value) => {
                                const scoreToRisk = { 1: "Low", 2: "Moderate", 3: "High" };
                                return scoreToRisk[value] || "";
                            }
                        }
                    }
                }
            },
        });

        await new Promise((r) => setTimeout(r, 50));
        const link = document.createElement("a");
        link.href = tempCanvas.toDataURL("image/png");
        link.download = filename;
        link.click();

        chart.destroy();
        document.body.removeChild(tempCanvas);
    }
});