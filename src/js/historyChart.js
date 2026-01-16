/**
 * class HistoryChart
 */
class HistoryChart {

    constructor(containerId) {
        this.container = d3.select(containerId);
        this.margin = { top: 10, right: 20, bottom: 40, left: 70 };

        // 1. Setup
        const height = this.container.node().clientHeight || 200;

        this.svg = this.container.append("svg")
            .attr("width", "100%")
            .attr("height", height)
            .style("overflow", "visible");

        this.g = this.svg.append("g")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`);

        this.width = this.container.node().clientWidth - this.margin.left - this.margin.right;
        this.height = height - this.margin.top - this.margin.bottom;

        // 2. Elements
        this.xAxisGroup = this.g.append("g");
        this.yAxisGroup = this.g.append("g");

        this.path = this.g.append("path")
            .attr("fill", "none")
            .attr("stroke", "steelblue")
            .attr("stroke-width", 2);

        this.yLabel = this.g.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", -45)
            .attr("x", -(this.height / 2))
            .style("text-anchor", "middle")
            .style("font-size", "12px")
            .style("fill", "#666");

        this.x = d3.scaleTime().range([0, this.width]);
        this.y = d3.scaleLinear().range([this.height, 0]);
    }

    /**
     * Updates the chart.
     */
    update(data, variable = "2t") {
        if (!data || data.length === 0) return;

        // 1. Dimensions
        this.width = this.container.node().clientWidth - this.margin.left - this.margin.right;
        this.x.range([0, this.width]);

        // 2. Process Data
        const nestedData = d3.groups(data, d => `${d.year}-${d.month}`);

        const processedData = nestedData.map(([key, values]) => {
            const [year, month] = key.split("-");
            return {
                date: new Date(year, month - 1),
                value: d3.mean(values, v => +v[variable])
            };
        }).sort((a, b) => a.date - b.date);

        // 3. Domains
        this.x.domain(d3.extent(processedData, d => d.date));

        const [min, max] = d3.extent(processedData, d => d.value);
        this.y.domain([min * 0.99, max * 1.01]);

        // 4. Draw
        this.xAxisGroup.attr("transform", `translate(0,${this.height})`)
            .call(d3.axisBottom(this.x).ticks(5));

        this.yAxisGroup.transition().duration(500)
            .call(d3.axisLeft(this.y));

        const lineGenerator = d3.line()
            .curve(d3.curveMonotoneX)
            .x(d => this.x(d.date))
            .y(d => this.y(d.value));

        this.path.datum(processedData)
            .transition().duration(500)
            .attr("d", lineGenerator);

        // 5. Labels
        const conf = window.climateVariables[variable] || {};
        const labelText = conf.label || variable;
        const unitText = conf.unit || "";
        this.yLabel.text(`${labelText} (${unitText})`);

        // 6. Interaction
        const dots = this.g.selectAll(".dot").data(processedData);

        dots.exit().remove();

        dots.enter().append("circle")
            .attr("class", "dot")
            .merge(dots)
            .attr("cx", d => this.x(d.date))
            .attr("cy", d => this.y(d.value))
            .attr("r", 5)
            .attr("fill", "steelblue")
            .attr("opacity", 0)

            .on("mouseover", function() {
                d3.select(this)
                    .attr("opacity", 1)
                    .attr("fill", "orange")
                    .attr("r", 7);
                this.style.cursor = "pointer";
            })
            .on("mouseout", function() {
                d3.select(this)
                    .attr("opacity", 0)
                    .attr("r", 5);
            })
            .on("click", (event, d) => {
                d3.selectAll(".dot").attr("opacity", 0);
                d3.select(event.currentTarget)
                    .attr("opacity", 1)
                    .attr("fill", "#e63946");

                window.dispatchEvent(new CustomEvent("dateChanged", {
                    detail: {
                        year: d.date.getFullYear(),
                        month: d.date.getMonth() + 1
                    }
                }));
            });
    }
}