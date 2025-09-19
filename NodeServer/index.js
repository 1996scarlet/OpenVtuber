var app = require('http').createServer(handler)
var io = require('socket.io')(app);
var fs = require('fs');

const path = require('path');

function handler(req, res) {
    const ROOT = __dirname; // Define the safe root directory
    let filePath;

    try {
        // Resolve and normalize the file path
        filePath = path.resolve(ROOT, '.' + req.url);
        filePath = fs.realpathSync(filePath);

        // Ensure the file path is within the root directory
        if (!filePath.startsWith(ROOT)) {
            res.writeHead(403);
            return res.end('Access denied');
        }
    } catch (err) {
        res.writeHead(400);
        return res.end('Invalid file path');
    }

    fs.readFile(filePath, function (err, data) {
        if (err) {
            res.writeHead(500);
            return res.end('Error loading file');
        }

        res.writeHead(200);
        res.end(data);
    });
}

io.of('/kizuna').on('connection', (socket) => {
    console.log('a kizuna client connected');

    socket.on('result_data', (result) => {
        if (result != 0) {
            socket.broadcast.emit('result_download', result);
        }
    });

    socket.on('disconnect', () => { console.log('a kizuna client disconnected') });
});

app.listen(6789, () => console.log('listening on http://127.0.0.1:6789/kizuna.html'));
