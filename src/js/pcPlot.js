/**
 * class ParallelCoordinates
 */
class ParallelCoordinates {

    constructor(containerId) {
        this.container = d3.select(containerId);
        this.margin = {top: 30, right: 40, bottom: 0, left: 20};

        // 1. Setup
        const height = this.container.node().clientHeight || 400;

        this.svg = this.container.append("svg")
            .attr("width", "100%")
            .attr("height", height)
            .style("overflow", "visible");

        this.g = this.svg.append("g")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`);

        this.pathGroup = this.g.append("g").attr("class", "paths");
        this.axisGroup = this.g.append("g").attr("class", "axes");

        this.yScales = {};
        this.xScale = d3.scalePoint().padding(0.2);
        this.metaKeys = ["year", "month", "lat", "lon", "country_code"];
    }

    /**
     * Updates the chart.
     */
    update(data) {
        if (!data || data.length === 0) return;

        // limit data so it can be displayed
        const limit = 100;
        const step = Math.ceil(data.length / limit);
        const displayData = data.filter((d, i) => i % step === 0);

        // 1. Dimensions
        const containerRect = this.container.node().getBoundingClientRect();
        this.width = containerRect.width - this.margin.left - this.margin.right;
        const height = containerRect.height - this.margin.top - this.margin.bottom;
        this.svg.attr("height", containerRect.height);

        const keys = Object.keys(displayData[0]);
        this.dimensions = keys.filter(d => !this.metaKeys.includes(d));

        // 2. Scales
        this.xScale.domain(this.dimensions).range([0, this.width]);
        this.dimensions.forEach(dim => {
            this.yScales[dim] = d3.scaleLinear()
                .domain(d3.extent(displayData, d => +d[dim]))
                .range([height, 0]);
        });

        // 3. Lines
        const lineGenerator = d3.line();
        const path = d => lineGenerator(this.dimensions.map(p =>
            [this.xScale(p), this.yScales[p](d[p])]
        ));

        this.pathGroup.selectAll("path")
            .data(displayData)
            .join("path")
            .attr("d", path)
            .style("fill", "none")
            .style("stroke", "#4682B4")
            .style("stroke-width", 1)
            .style("opacity", 0.25);

        // 4. Axes
        this.axisGroup.selectAll(".dimension").remove();

        const axes = this.axisGroup.selectAll(".dimension")
            .data(this.dimensions)
            .enter().append("g")
            .attr("class", "dimension")
            .attr("transform", d => `translate(${this.xScale(d)})`);

        axes.each((d, i, nodes) => {
            d3.select(nodes[i])
                .call(d3.axisLeft(this.yScales[d])
                    .ticks(5)
                    .tickFormat(d3.format(".3~g"))
                );
        });

        axes.selectAll("text")
            .style("fill", "#444")
            .style("font-size", "9px")
            .style("text-shadow", "0 1px 0 #fff, 1px 0 0 #fff, 0 -1px 0 #fff, -1px 0 0 #fff");

        axes.append("text")
            .style("text-anchor", "middle")
            .attr("y", -10)
            .text(d => {
                const conf = (window.climateVariables && window.climateVariables[d])
                    ? window.climateVariables[d]
                    : null;
                return conf ? `${conf.label} (${conf.unit})` : d;
            })
            .style("fill", "black")
            .style("font-size", "10px")
            .style("font-weight", "bold")
            .style("cursor", "default")
            .style("text-shadow", "0 1px 0 #fff, 1px 0 0 #fff, 0 -1px 0 #fff, -1px 0 0 #fff");

        // 5. Brushing
        const yScales = this.yScales;
        const pathGroup = this.pathGroup;

        axes.append("g")
            .attr("class", "brush")
            .each(function (d) {
                d3.select(this).call(
                    d3.brushY()
                        .extent([[-10, 0], [10, height]])
                        .on("brush end", function (event) {
                            const selection = event.selection;

                            if (!selection) {
                                pathGroup.selectAll("path").style("display", null);
                                return;
                            }

                            const [y1, y0] = selection.map(yScales[d].invert);
                            const minVal = Math.min(y1, y0);
                            const maxVal = Math.max(y1, y0);

                            pathGroup.selectAll("path")
                                .style("display", row => {
                                    const val = +row[d];
                                    return (val >= minVal && val <= maxVal) ? null : "none";
                                });
                        })
                );
            });
    }
}