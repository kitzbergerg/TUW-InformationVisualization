/**
 * Global State & Configuration
 */
window.appState = {
    selectedCountry: null,
    activeVariable: "2t",
    fullCountryData: [],
    globalData: [],
    selectedYear: null,
    selectedMonth: null
};

const variableFiles = {
    "2t": "https://media.githubusercontent.com/media/kitzbergerg/TUW-InformationVisualization/main/src/data/global/country_avg_2t.csv",
    "tp": "https://media.githubusercontent.com/media/kitzbergerg/TUW-InformationVisualization/main/src/data/global/country_avg_tp.csv",
    "10si": "https://media.githubusercontent.com/media/kitzbergerg/TUW-InformationVisualization/main/src/data/global/country_avg_10si.csv",
    "2d": "https://media.githubusercontent.com/media/kitzbergerg/TUW-InformationVisualization/main/src/data/global/country_avg_2d.csv",
    "swvl1": "https://media.githubusercontent.com/media/kitzbergerg/TUW-InformationVisualization/main/src/data/global/country_avg_swvl1.csv",
    "sde": "https://media.githubusercontent.com/media/kitzbergerg/TUW-InformationVisualization/main/src/data/global/country_avg_sde.csv",
    "sf": "https://media.githubusercontent.com/media/kitzbergerg/TUW-InformationVisualization/main/src/data/global/country_avg_sf.csv",
    "skt": "https://media.githubusercontent.com/media/kitzbergerg/TUW-InformationVisualization/main/src/data/global/country_avg_skt.csv",
    "ssr": "https://media.githubusercontent.com/media/kitzbergerg/TUW-InformationVisualization/main/src/data/global/country_avg_ssr.csv",
    "slhf": "https://media.githubusercontent.com/media/kitzbergerg/TUW-InformationVisualization/main/src/data/global/country_avg_slhf.csv",
    "sshf": "https://media.githubusercontent.com/media/kitzbergerg/TUW-InformationVisualization/main/src/data/global/country_avg_sshf.csv"
};

const customBlue = d3.interpolateRgb("#6baed6", "#08306b");
const customTeal = d3.interpolateRgb("#4db6ac", "#004d40");

window.climateVariables = {
    "2t":   { label: "Temperature (2m)",      unit: "K",      color: d3.interpolateRdYlBu, reverse: true },
    "2d":   { label: "Dewpoint Temp",         unit: "K",      color: d3.interpolateRdYlBu, reverse: true },
    "skt":  { label: "Skin Temp",             unit: "K",      color: d3.interpolateRdYlBu, reverse: true },
    "tp":   { label: "Total Precipitation",   unit: "m",      color: customBlue,  reverse: false },
    "sde":  { label: "Snow Depth",            unit: "m",      color: customBlue,   reverse: false },
    "sf":   { label: "Snowfall",              unit: "m",      color: customBlue,   reverse: false },
    "swvl1":{ label: "Soil Water",            unit: "m³/m³",  color: customTeal,   reverse: false },
    "10si": { label: "Wind Speed",            unit: "m/s",    color: d3.interpolateViridis,reverse: false },
    "ssr":  { label: "Solar Radiation",       unit: "J/m²",   color: d3.interpolateInferno,reverse: false },
    "slhf": { label: "Latent Heat Flux",      unit: "J/m²",   color: d3.interpolateMagma,  reverse: false },
    "sshf": { label: "Sensible Heat Flux",    unit: "J/m²",   color: d3.interpolateMagma,  reverse: false }
};

window.climateVars = {};
Object.keys(window.climateVariables).forEach(key => window.climateVars[key] = window.climateVariables[key].label);
window.variableUnits = {};
Object.keys(window.climateVariables).forEach(key => window.variableUnits[key] = window.climateVariables[key].unit);

const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
let map, pcPlot, historyChart;

/**
 * Creates the Radio Button Panel.
 */
function createFilterPanel() {
    const container = d3.select("#variable-checkboxes");
    container.html("");

    Object.keys(window.climateVariables).forEach((key) => {
        const conf = window.climateVariables[key];
        const div = container.append("div").attr("class", "filter-item");

        div.append("input")
            .attr("type", "radio")
            .attr("name", "climate-variable")
            .attr("id", "chk-" + key)
            .attr("value", key)
            .property("checked", key === "2t")
            .on("change", function () {
                updateMapVariable(key);
            });

        div.append("label").attr("for", "chk-" + key).text(" " + conf.label);
    });
}

/**
 * Creates Time Filter Dropdowns.
 */
function createTimeFilters(data) {
    const yearSelect = d3.select("#year-select");
    const monthSelect = d3.select("#month-select");
    const years = [...new Set(data.map(d => +d.year))].sort((a, b) => b - a);

    yearSelect.html("");
    years.forEach(y => yearSelect.append("option").attr("value", y).text(y));

    monthSelect.html("");
    monthNames.forEach((name, index) => {
        monthSelect.append("option").attr("value", index + 1).text(name);
    });

    if (!appState.selectedYear) appState.selectedYear = years[0];
    if (!appState.selectedMonth) appState.selectedMonth = 1;

    yearSelect.property("value", appState.selectedYear);
    monthSelect.property("value", appState.selectedMonth);

    yearSelect.on("change", function() {
        appState.selectedYear = +this.value;
        filterAllViewsByTime();
    });
    monthSelect.on("change", function() {
        appState.selectedMonth = +this.value;
        filterAllViewsByTime();
    });
}

/**
 * Creates Country Filter Dropdown with Map Sync.
 */
function createCountryFilter(geoData) {
    const select = d3.select("#country-select");
    select.html("");

    select.append("option").attr("value", "GLOBAL").text("🌍 Global View");

    const countries = geoData.features.map(d => ({
        id: d.properties.ADM0_A3 || d.properties.ISO_A3 || d.properties.iso_a3 || d.id,
        name: d.properties.NAME || d.properties.name || d.properties.admin
    })).filter(c => c.id && c.name)
      .sort((a, b) => a.name.localeCompare(b.name));

    countries.forEach(c => {
        select.append("option").attr("value", c.id).text(c.name);
    });

    select.on("change", function() {
        const val = this.value;
        if (val === "GLOBAL") {
            if (map) map.reset();
        } else {
            if (map) map.selectCountry(val);
        }
    });
}

/**
 * Updates all views based on time.
 */
function filterAllViewsByTime() {
    console.log(`Global Update: ${appState.selectedYear}-${appState.selectedMonth}`);

    if (map && appState.globalData.length > 0) {
        const globalFiltered = appState.globalData.filter(d =>
            +d.year === appState.selectedYear && +d.month === appState.selectedMonth
        );
        map.avgData = globalFiltered;
        map.renderWorld();
    }

    if (appState.selectedCountry && appState.fullCountryData.length > 0) {
        const countryFiltered = appState.fullCountryData.filter(d =>
            +d.year === appState.selectedYear &&
            +d.month === appState.selectedMonth
        );

        if (map) map.renderDetailedGrid(countryFiltered, appState.activeVariable);
        if (pcPlot) pcPlot.update(countryFiltered);
    }
}

/**
 * Switches the active variable.
 */
function updateMapVariable(variableKey) {
    console.log(`Switching Variable to: ${variableKey}`);
    d3.select("#current-var-display").text(window.climateVariables[variableKey].label);

    d3.csv(variableFiles[variableKey]).then(data => {
        appState.activeVariable = variableKey;
        appState.globalData = data;

        filterAllViewsByTime();

        if (appState.selectedCountry && appState.fullCountryData.length > 0) {
            const countryFiltered = appState.fullCountryData.filter(d =>
                +d.year === appState.selectedYear &&
                +d.month === appState.selectedMonth
            );
            if (map) map.renderDetailedGrid(countryFiltered, appState.activeVariable);
            if (historyChart) historyChart.update(appState.fullCountryData, appState.activeVariable);
        }
    }).catch(err => console.error("Could not load file:", variableFiles[variableKey]));
}

/**
 * Initialization.
 */
Promise.all([
    d3.json("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"),
    d3.csv(variableFiles["2t"])
]).then(([geoData, avgData]) => {

    appState.globalData = avgData;
    createFilterPanel();
    createTimeFilters(avgData);
    createCountryFilter(geoData);

    map = new WorldMap("#map-container", geoData, avgData);
    filterAllViewsByTime();

    if (typeof ParallelCoordinates !== 'undefined') pcPlot = new ParallelCoordinates("#pc-container");
    if (typeof HistoryChart !== 'undefined') historyChart = new HistoryChart("#line-container");

    d3.select("#history-panel").style("opacity", "0").style("pointer-events", "none");

    // 1. Country Selected
    window.addEventListener('countrySelected', (e) => {
        appState.selectedCountry = e.detail;

        d3.select("#country-select").property("value", appState.selectedCountry);
        d3.select("#main-grid").classed("details-active", true);
        d3.select("#history-panel").style("opacity", "1").style("pointer-events", "all");

        const csvPath = `https://media.githubusercontent.com/media/kitzbergerg/TUW-InformationVisualization/main/src/data/countries/era5_monthly_${appState.selectedCountry}.csv`;

        d3.csv(csvPath).then(data => {
            appState.fullCountryData = data;

            if (historyChart) historyChart.update(data, appState.activeVariable);
            filterAllViewsByTime();

            setTimeout(() => {
                 const currentData = data.filter(d =>
                    +d.year === appState.selectedYear && +d.month === appState.selectedMonth
                 );
                 if (pcPlot) pcPlot.update(currentData);
            }, 550);
        }).catch(err => console.error("Error loading country details", err));
    });

    // 2. Map Reset
    window.addEventListener('mapReset', () => {
        console.log("Map Reset: Returning to Global View");
        appState.selectedCountry = null;
        appState.fullCountryData = [];

        d3.select("#country-select").property("value", "GLOBAL");
        d3.select("#main-grid").classed("details-active", false);
        d3.select("#history-panel").style("opacity", "0").style("pointer-events", "none");

        if (historyChart) {
             d3.select("#line-container").selectAll("*").remove();
             historyChart = new HistoryChart("#line-container");
        }
    });

    // 3. Date Changed
    window.addEventListener('dateChanged', (e) => {
        const { year, month } = e.detail;
        appState.selectedYear = +year;
        appState.selectedMonth = +month;

        d3.select("#year-select").property("value", appState.selectedYear);
        d3.select("#month-select").property("value", appState.selectedMonth);

        filterAllViewsByTime();
    });

}).catch(err => console.error("Initialization error:", err));