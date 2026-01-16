/**
 * class WorldMap
 */
class WorldMap {

    constructor(containerId, geoData, initialAvgData) {
        this.container = d3.select(containerId);
        this.geoData = geoData;
        this.avgData = initialAvgData;
        this.width = this.container.node().getBoundingClientRect().width;
        this.height = this.container.node().clientHeight || 600;
        this.selectedCountry = null;
        this.init();
    }

    init() {
        this.svg = this.container.append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("viewBox", `0 0 ${this.width} ${this.height}`)
            .attr("preserveAspectRatio", "xMidYMid meet");

        // 1. Background Click
        this.svg.on("click", (event) => {
            if (event.target.tagName === 'svg' || event.target.tagName === 'g') {
                this.reset();
            }
        });

        // 2. Layers
        this.g = this.svg.append("g").attr("class", "map-layer");
        this.gridGroup = this.svg.append("g").attr("class", "grid-layer");
        this.borderGroup = this.svg.append("g").attr("class", "border-layer");

        this.legendG = this.svg.append("g")
            .attr("class", "map-legend")
            .attr("transform", `translate(20, ${this.height - 50})`);

        this.defs = this.svg.append("defs");
        this.clipPath = this.defs.append("clipPath")
            .attr("id", "selected-country-clip")
            .append("path");

        // 3. Projection
        this.projection = d3.geoNaturalEarth1()
            .scale(this.width / 1.6 / Math.PI)
            .translate([this.width / 2, this.height / 2]);
        this.path = d3.geoPath().projection(this.projection);

        this.colorScale = d3.scaleSequential();

        this.tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);

        this.zoom = d3.zoom()
            .scaleExtent([1, 8])
            .on("zoom", (event) => {
                const t = event.transform;
                this.g.attr("transform", t);
                this.gridGroup.attr("transform", t);
                this.borderGroup.attr("transform", t);
                this.g.selectAll("path").attr("stroke-width", 0.5 / t.k);
                this.borderGroup.selectAll("path").attr("stroke-width", 1.5 / t.k);
            });

        this.svg.call(this.zoom);
        this.renderWorld();
    }

    /**
     * Helper to get Country ID.
     */
    getCountryId(d) {
        return d.properties.ADM0_A3 || d.properties.ISO_A3 || d.properties.iso_a3 || d.id;
    }

    /**
     * External selection method.
     */
    selectCountry(id) {
        const feature = this.geoData.features.find(d => this.getCountryId(d) === id);

        if (feature) {
            if (this.selectedCountry === feature) return;

            this.selectedCountry = feature;
            const [[x0, y0], [x1, y1]] = this.path.bounds(feature);

            this.svg.transition().duration(750).call(
                this.zoom.transform,
                d3.zoomIdentity.translate(this.width / 2, this.height / 2)
                    .scale(Math.min(8, 0.9 / Math.max((x1 - x0) / this.width, (y1 - y0) / this.height)))
                    .translate(-(x0 + x1) / 2, -(y0 + y1) / 2)
            );

            window.dispatchEvent(new CustomEvent("countrySelected", { detail: id }));
        }
    }

    /**
     * Renders global map.
     */
    renderWorld() {
        const activeKey = window.appState.activeVariable;
        const conf = window.climateVariables[activeKey];
        const values = this.avgData.map(d => +d.value);
        let minVal = d3.min(values) || 0;
        let maxVal = d3.max(values) || 1;

        this.colorScale.interpolator(conf.color);
        if (conf.reverse) this.colorScale.domain([maxVal, minVal]);
        else this.colorScale.domain([minVal, maxVal]);

        this.gridGroup.selectAll("*").remove();
        this.borderGroup.selectAll("*").remove();

        this.g.selectAll(".country")
            .data(this.geoData.features)
            .join("path")
            .attr("class", "country")
            .attr("d", this.path)
            .attr("stroke", "#fff")
            .attr("stroke-width", 0.5)
            .on("mouseover", (event, d) => this.showTooltip(event, d))
            .on("mouseout", () => this.hideTooltip())
            .on("click", (event, d) => this.clicked(event, d))
            .transition().duration(750)
            .attr("fill", d => {
                const code = this.getCountryId(d);
                const record = this.avgData.find(r => r.country_code === code);
                return record ? this.colorScale(+record.value) : "#e0e0e0";
            });

        this.drawLegend(minVal, maxVal, conf.unit, conf.color, conf.reverse);
    }

    /**
     * Renders detailed grid.
     */
    renderDetailedGrid(detailData, variable = '2t') {
        this.gridGroup.selectAll("*").remove();
        this.borderGroup.selectAll("*").remove();

        if (!detailData || detailData.length === 0 || !this.selectedCountry) return;

        const conf = window.climateVariables[variable];
        const extent = d3.extent(detailData, d => +d[variable]);
        const localScale = d3.scaleSequential(conf.color);
        if (conf.reverse) localScale.domain([extent[1], extent[0]]);
        else localScale.domain([extent[0], extent[1]]);

        this.clipPath.attr("d", this.path(this.selectedCountry));

        const cellSize = 5;
        const cellGroup = this.gridGroup.append("g").attr("clip-path", "url(#selected-country-clip)");

        cellGroup.selectAll(".grid-cell")
            .data(detailData)
            .enter()
            .append("rect")
            .attr("class", "grid-cell")
            .attr("x", d => this.projection([+d.lon, +d.lat])[0] - cellSize / 2)
            .attr("y", d => this.projection([+d.lon, +d.lat])[1] - cellSize / 2)
            .attr("width", cellSize)
            .attr("height", cellSize)
            .attr("fill", d => localScale(+d[variable]))
            .on("mouseover", (event, d) => {
                this.tooltip.transition().duration(100).style("opacity", .9);
                this.tooltip.html(`
                    <strong>${variable}:</strong> ${(+d[variable]).toFixed(2)} ${conf.unit}<br/>
                    <strong>Loc:</strong> ${d.lat}, ${d.lon}
                `).style("left", (event.pageX + 10) + "px").style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", () => {
                this.tooltip.transition().duration(500).style("opacity", 0);
            })
            .attr("opacity", 0)
            .transition().duration(1000)
            .attr("opacity", 1);

        this.borderGroup.append("path")
            .datum(this.selectedCountry)
            .attr("d", this.path)
            .attr("fill", "none")
            .attr("stroke", "#333")
            .attr("stroke-width", 1.5)
            .style("pointer-events", "none");

        this.drawLegend(extent[0], extent[1], conf.unit, conf.color, conf.reverse);
    }

    /**
     * Draws Legend.
     */
    drawLegend(min, max, unit, interpolator, isReversed) {
        this.legendG.html("");
        const legendWidth = 200;
        const legendHeight = 12;

        const defs = this.legendG.append("defs");
        const gradientId = "legend-gradient";
        const linearGradient = defs.append("linearGradient")
            .attr("id", gradientId).attr("x1", "0%").attr("y1", "0%").attr("x2", "100%").attr("y2", "0%");

        const scale = d3.scaleSequential(interpolator);

        if (isReversed) {
            linearGradient.append("stop").attr("offset", "0%").attr("stop-color", scale(1));
            linearGradient.append("stop").attr("offset", "100%").attr("stop-color", scale(0));
        } else {
            linearGradient.append("stop").attr("offset", "0%").attr("stop-color", scale(0));
            linearGradient.append("stop").attr("offset", "100%").attr("stop-color", scale(1));
        }

        this.legendG.append("rect").attr("width", legendWidth).attr("height", legendHeight).style("fill", `url(#${gradientId})`).attr("stroke", "#ccc");
        this.legendG.append("text").attr("x", 0).attr("y", legendHeight + 15).text(`${min.toFixed(1)} ${unit}`).style("font-size", "11px").style("fill", "#333");
        this.legendG.append("text").attr("x", legendWidth).attr("y", legendHeight + 15).attr("text-anchor", "end").text(`${max.toFixed(1)} ${unit}`).style("font-size", "11px").style("fill", "#333");
        this.legendG.append("text").attr("x", 0).attr("y", -6).text(`Range (${unit})`).style("font-size", "11px").style("font-weight", "bold").style("fill", "#555");
    }

    /**
     * Tooltip.
     */
    showTooltip(event, d) {
        const code = this.getCountryId(d);
        const record = this.avgData.find(r => r.country_code === code);
        const val = record ? +record.value : null;
        const displayVal = val !== null ? val.toFixed(2) : 'N/A';
        const unit = (window.climateVariables[window.appState.activeVariable] || {}).unit || "";

        this.tooltip.transition().duration(200).style("opacity", .9);
        this.tooltip.html(`<strong>${d.properties.name || d.properties.NAME}</strong><br/>Val: ${displayVal} ${unit}`)
            .style("left", (event.pageX + 10) + "px").style("top", (event.pageY - 28) + "px");
    }

    hideTooltip() {
        this.tooltip.transition().duration(500).style("opacity", 0);
    }

    /**
     * Click Interaction.
     */
    clicked(event, d) {
        const code = this.getCountryId(d);
        const record = this.avgData.find(r => r.country_code === code);

        if (!record) return;

        if (this.selectedCountry === d) return this.reset();

        this.selectedCountry = d;
        const [[x0, y0], [x1, y1]] = this.path.bounds(d);
        event.stopPropagation();

        this.svg.transition().duration(750).call(
            this.zoom.transform,
            d3.zoomIdentity.translate(this.width / 2, this.height / 2)
                .scale(Math.min(8, 0.9 / Math.max((x1 - x0) / this.width, (y1 - y0) / this.height)))
                .translate(-(x0 + x1) / 2, -(y0 + y1) / 2)
        );

        window.dispatchEvent(new CustomEvent("countrySelected", { detail: code }));
    }

    /**
     * Reset Map.
     */
    reset() {
        this.selectedCountry = null;
        this.svg.transition().duration(750).call(this.zoom.transform, d3.zoomIdentity);
        this.gridGroup.selectAll("*").remove();
        this.borderGroup.selectAll("*").remove();
        window.dispatchEvent(new CustomEvent("mapReset"));
    }
}