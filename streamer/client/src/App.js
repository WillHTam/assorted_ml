import React, { Component } from 'react'
import './App.css'
import axios from 'axios'
import moment from 'moment'

import GitHubForkRibbon from 'react-github-fork-ribbon'
import { LinearGradient } from '@vx/gradient'
import { withScreenSize } from '@vx/responsive'

import Chart from './components/chart'
import formatPrice from './components/formatPrice'

function Background({ width, height }) {
  return <svg width={width} height={height}>
    <LinearGradient id="fill" vertical={false}>
      <stop stopColor="#2c4e99" offset="0%" />
      <stop stopColor="#6e89d8" offset="45%" />
      <stop stopColor="#FFAF84" offset="100%" />
    </LinearGradient>
    <rect width={width} height={height} fill="url(#fill)" />
  </svg>
}

class App extends Component {
  constructor(props) {
    super(props)
    this.state = {
      data: {}
    }
  }
  
  componentWillMount() {
    fetch('https://api.coindesk.com/v1/bpi/historical/close.json')
      .then(res => res.json())
      .then(json => {
        this.setState({
          data: json
        })
      })
  }
  
  render() {
    const screenWidth = window.innerWidth
    const screenHeight = window.innerHeight
    const { data } = this.state

    if (!data.bpi) return <div className="main"><h2>Loading...</h2></div>
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

    return (
      <div className="App">
        <GitHubForkRibbon href="https://www.github.com/willhtam"
          target="_blank"
          position="right"
          color="orange"
          >
          Fork me on GitHub!
        </GitHubForkRibbon>

        <Background width={screenWidth} height={screenHeight} />
        <div className="main">
          <div className="chart">
            <div className="titleBar">
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
          <h1>Hellou</h1>
        </div>
      </div>
    )
  }
}

export default withScreenSize(App)
