const express = require('express');
var app = express();
var server = require('http').createServer(app);
var io = require('socket.io')(server);

app.get('/vtuber', (req, res) => res.sendFile(__dirname + '/simple.html'));
app.use(express.static('public'));

io.of('/sakuya').on('connection', (socket) => {
    console.log('a vtuber client connected');

    socket.on('frame_data', (msg) => {
        socket.emit('response', '');
        if (msg != 0) {
            socket.broadcast.emit('frame_download', msg.toString('base64'));
        }
    });

    socket.on('result_data', (result) => {
        // console.log(result);
        socket.broadcast.emit('result_download', result);
    });

    socket.on('disconnect', () => { console.log('a vtuber client disconnected') });
});

server.listen(6789, () => console.log('listening on *:6789'));
