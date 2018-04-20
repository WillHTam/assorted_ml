var datum = ''

function update(req, res, next) {
    const time = req.body.timeActivation
    const value = req.body.currentValue
    datum = req.body.currentValue

    console.log(time, value)
    return res.status(200).json({ message: 'goud' })
}

function pass(req, res, next) {
    if ( datum == '' ) {
        return res.status(200).json({ message: 'SERVER DOWN' })
    } else {
        return res.status(200).json({ message: datum })
    }
}

module.exports = {
    update,
    pass
}