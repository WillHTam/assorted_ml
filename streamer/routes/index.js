var express = require('express')
var router = express.Router()
var bodyParser = require('body-parser')
var mainController = require('../controllers/mainController')
const path = require('path')

router.use(bodyParser.json())
router.use(bodyParser.urlencoded({ extended: true }))

router.get("/", function (req, res, next) {
    res.use(express.static(path.join(__dirname, './client/build')) )
})

router.post('/post', mainController.update)

module.exports = router