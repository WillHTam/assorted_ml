import React, { Component } from 'react'
import { withParentSize } from "@vx/responsive"
import { scaleTime, scaleLinear } from "@vx/scale"
import { LinePath, AreaClosed, Bar, Line } from "@vx/shape"
import { LinearGradient } from "@vx/gradient"
import { AxisBottom } from "@vx/axis"
import { withTooltip, Tooltip } from "@vx/tooltip"
import { localPoint } from '@vx/event'
import { bisector } from 'd3-array'

import formatPrice from "./formatPrice"
import formatDate from "./formatDate"
import MaxPrice from "./maxprice"
import MinPrice from "./minprice"

class Chart extends React.Component {
    constructor(props) {
        super(props)
    }
    render() {
        const { data, parentWidth, parentHeight, tooltipLeft, tooltipTop, tooltipData, showTooltip, hideTooltip } = this.props
        const margin = {
            top: 15,
            bottom: 40,
            left: 0,
            right: 0
        }
        const width = parentWidth - margin.left - margin.right
        const height = parentHeight - margin.top - margin.bottom


        const x = d => new Date(d.time)
        const y = d => d.price
        const bisectDate = bisector(d => x(d)).left

        const firstPoint = data[0]
        const currentPoint = data[data.length - 1]

        const minPrice = Math.min(...data.map(y))
        const maxPrice = Math.max(...data.map(y))
        const maxPriceData = [
            { time: x(firstPoint), price: maxPrice },
            { time: x(currentPoint), price: maxPrice }
        ]
        const minPriceData = [
            { time: x(firstPoint), price: minPrice },
            { time: x(currentPoint), price: minPrice }
        ]
        // console.log("maxPriceData" + maxPriceData)

        const xScale = scaleTime({
            range: [0, width],
            domain: [Math.min(...data.map(x)), Math.max(...data.map(x))]
        })
        // console.log("X axis domain: " + xScale.domain())

        const yScale = scaleLinear({
            // Y increases as you go down so need to remap with range
            range: [height, 0],
            domain: [minPrice, maxPrice]
        })
        // console.log("Y axis domain: " + yScale.domain())

        return <div>
            <div>
                <title>Coin Prices</title>
                <meta name="viewport" content="initial-scale=1.0, width=device-width" />
            </div>
            <svg ref={s => (this.svg = s)} width={width} height={parentHeight}>
                <AxisBottom top={yScale(minPrice)} data={data} scale={xScale} x={x} numTicks={5} tickLabelComponent={<text fill="#ffffff" fontSize={11} textAnchor="middle" />} hideAxisLine hideTicks />
                <LinearGradient id="area-fill" from="#4682b4" to="#4682b4" fromOpacity={0.4} toOpacity={0} />
                <MaxPrice data={maxPriceData} yScale={yScale} xScale={xScale} x={x} y={y} label={formatPrice(maxPrice)} yText={yScale(maxPrice)} />
                <MinPrice data={minPriceData} yScale={yScale} xScale={xScale} x={x} y={y} label={formatPrice(minPrice)} yText={yScale(minPrice)} />
                <AreaClosed data={data} yScale={yScale} xScale={xScale} x={x} y={y} fill="url(#area-fill)" stroke="transparent" />
                <LinePath data={data} yScale={yScale} xScale={xScale} x={x} y={y} />
                <Bar data={data} width={width} height={height} fill="transparent" onMouseMove={data => event => {
                    // Convert event coordinates to local coordinates
                    const { x: xPoint } = localPoint(this.svg, event);
                    const x0 = xScale.invert(xPoint);
                    const index = bisectDate(data, x0, 1);
                    const d0 = data[index - 1];
                    const d1 = data[index];
                    const d = x0 - xScale(x(d0)) > xScale(x(d1)) - x0 ? d1 : d0;
                    showTooltip({
                        tooltipLeft: xScale(x(d)),
                        tooltipTop: yScale(y(d)),
                        tooltipData: d
                    });
                }} onMouseLeave={data => event => hideTooltip()} />
                {tooltipData && <g>
                    <Line from={{ x: tooltipLeft, y: yScale(y(maxPriceData[0])) }} to={{ x: tooltipLeft, y: yScale(y(minPriceData[0])) }} stroke="#ffffff" strokeDasharray="2,2" />
                    <circle r="8" cx={tooltipLeft} cy={tooltipTop} fill="#00f1a1" fillOpacity={0.4} style={{ pointerEvents: 'none' }} />
                    <circle r="4" cx={tooltipLeft} cy={tooltipTop} fill="#00f1a1" style={{ pointerEvents: 'none' }} />
                </g>}
            </svg>
            {tooltipData &&
                <div>
                    <Tooltip top={tooltipTop - 12} left={tooltipLeft - 12} style={{ backgroundColor: '#6086d6', color: '#ffffff' }}>
                        {formatPrice(y(tooltipData))}
                    </Tooltip>
                    <Tooltip left={tooltipLeft} top={yScale(minPrice)} style={{ transform: 'translateX(-50%)' }}>
                        {formatDate(x(tooltipData))}
                    </Tooltip>
                </div>
            }
        </div>
    }
}

export default withParentSize(withTooltip(Chart))
