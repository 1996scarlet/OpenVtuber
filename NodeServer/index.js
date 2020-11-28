const express = require('express');
var app = express();
var server = require('http').createServer(app);
var io = require('socket.io')(server);

app.get('/kizuna', (req, res) => res.sendFile(__dirname + '/kizuna.html'));
app.use(express.static('public'));


io.of('/kizuna').on('connection', (socket) => {
    console.log('a kizuna client connected');

    socket.on('result_data', (result) => {
        if (result != 0) {
            // socket.broadcast.emit('result_download', {
            //     'frame': result.frame.toString('base64'),
            //     'landmarks': result.landmarks,
            //     'euler': result.euler,
            //     'blink': result.blink,
            //     'points': result.points
            // });
            socket.broadcast.emit('result_download', result);
            // console.log(result);
        }
    });

    socket.on('disconnect', () => { console.log('a kizuna client disconnected') });
});

server.listen(6789, () => console.log('listening on http://127.0.0.1:6789/kizuna'));
