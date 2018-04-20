const http = require('http')
const express = require('express')
const bodyParser = require('body-parser')
const logger = require('morgan')
const port = process.env.PORT || 3000
const path = require('path')

const mainController = require('./controllers/mainController')

const app = express()

app.use(logger('dev'))
app.use(bodyParser.json())
app.use(bodyParser.urlencoded({ extended: true }))

app.use( function(req,res,next) {
    res.header("Access-Control-Allow-Origin", "*")
    res.header(
        "Access-Control-Allow-Headers",
        "Origin, X-Requested-With, Content-Type, Accept, email, auth_token, id"
    )
    res.header("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
    next()
})

app.use('/post', mainController.update)
app.use('/get', mainController.pass)
app.use('/', express.static(path.join(__dirname, './client/build')))

app.listen(port, () => {
    console.log(`Connected to ${port}`)
})

module.exports = app