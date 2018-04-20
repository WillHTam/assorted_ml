import { withScreenSize } from '@vx/responsive'
import { LinearGradient } from '@vx/gradient'
import Chart from '../components/chart'
import formatPrice from '../utils/formatPrice'

function Background({ width, height }) {
    return <svg width={width} height={height}>
        <LinearGradient id="fill" vertical={false}>
            <stop stopColor="#2c4e99" offset="0%" />
            <stop stopColor="#6e89d8" offset="45%" />
            <stop stopColor="#FFAF84" offset="100%" />
        </LinearGradient>
        <rect width={width} height={height} fill="url(#fill)" />
    </svg>;
}

class App extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            data: {}
        }
    }

    componentDidMount() {
        fetch('https://api.coindesk.com/v1/bpi/historical/close.json')
            .then(res => res.json())
            .then(json => {
                this.setState({
                    data: json
                })
            })
    }

    render() {
        const { screenWidth, screenHeight } = this.props
        const { data } = this.state

        if (!data.bpi) return <div className="loader"><h2>Loading...</h2></div>
        const prices = Object.keys(data.bpi).map(k => {
            return {
                time: k,
                price: data.bpi[k]
            }
        })

        const currentPrice = prices[prices.length - 1].price
        const lastPrice = prices[prices.length - 2].price
        const diffPrice = currentPrice - lastPrice
        const hasIncreased = diffPrice > 0

        console.log('response: ' + data)
        console.log('prices: ' + prices)
        console.log('current price: ' + currentPrice)

        return <div className="app">
            <Background width={screenWidth} height={screenHeight} />
            <div className="chartarea">
                <div className="chart">
                    <div className="titlebar">
                        <div className="title">
                            <div>BTC Price</div>
                            <div className="thirtee">
                                <small>Last 30 Days</small>
                            </div>
                        </div>
                        <div className="spacer" />
                        <div className="prices">
                            <div>
                                {formatPrice(currentPrice)}
                            </div>
                            <div className={hasIncreased ? 'increased' : 'decreased'}>
                                <small>
                                    {hasIncreased ? '+' : '-'}{formatPrice(diffPrice)}
                                </small>
                            </div>
                        </div>
                        <div />
                    </div>
                    <div className="container">
                        <Chart data={prices} />
                    </div>
                </div>
                <p className="disclaimer">
                    {data.disclaimer}   <a href="https://www.coindesk.com/price/">Powered by CoinDesk</a>
                </p>
            </div>
            <style jsx>{`.app,
              .chartarea {
                display: flex
                position: absolute
                top: 0
                left: 0
                right: 0
                bottom: 0
                flex: 1
                justify-content: center
                align-items: center
                font-family: arial
                flex-direction: column
              }
              .title {
              }
              .thirtee {
                  color: steelblue
              }
              .prices {
                align-items: flex-end
                display: flex
                flex-direction: column
              }
              .increased {
                color: #00f1a1
              }
              .decreased {
                color: red 
              }
              .spacer {
                flex: 1
              }
              .titlebar {
                display: flex
                flex-direction: row
                align-items: center
                padding: 15px
              }
              .container {
                flex: 1
                display: flex
              }
              .chart {
                width: 800px
                height: 500px
                background-color: #27273f
                color: white
                border-radius: 8px
                display: flex
                flex-direction: column
              }
              .disclaimer {
                color: white
                opacity: 0.4
                font-size: 11px
              }`}</style>
        </div>
    }
}

export default withScreenSize(App)