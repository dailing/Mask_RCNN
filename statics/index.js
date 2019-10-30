Vue.use(AsyncComputed)


class Box {
    constructor(){
        this.x=0;
        this.width=0;

        this.y=0;
        this.height=0;
        this.class = -1;

        
        this.dx=0;
        this.dy=0;
        
        this.status=0;
        // 0: not selected
        // 10: moving status
        // 20: resizing status

        this.boder_range = 5;
        this.event = null;
        this.mouse_down_event = null;
        this.record = null;
    }

    get top_left() {
        if (this.status == 0) {
            return [this.x - this.width / 2,
                    this.y - this.height / 2,
                    this.width,
                    this.height]
        } else if (this.status == 10) {
            var dx = this.mouse_down_event.offsetX - this.event.offsetX;
            var dy = this.mouse_down_event.offsetY - this.event.offsetY;
            return [this.x - this.width / 2 - dx,
                this.y - this.height / 2 - dy,
                this.width,
                this.height]
        } else if (this.status == 20) {
            var dx = this.event.offsetX - this.mouse_down_event.offsetX;
            var dy = this.event.offsetY - this.mouse_down_event.offsetY;
            var p1 = [this.x - this.width / 2, this.y - this.height / 2];
            var p2 = [this.x + this.width / 2, this.y + this.height / 2];
            p2[0] += dx;
            p2[1] += dy;
            return [Math.min(p1[0], p2[0]),
                    Math.min(p1[1], p2[1]),
                    Math.abs(p1[0] - p2[0]),
                    Math.abs(p1[1] - p2[1])]
        }
    }

    get bbox() {
        return [this.x, this.y, this.width, this.height];
    }

    get _in() {
        var x = event.offsetX;
        var y = event.offsetY;
        return (this.width / 2 - Math.abs(this.x - x) > 0) &&
                (this.height / 2 - Math.abs(this.y - y) > 0);
    }

    get _boder() {
        var x = event.offsetX;
        var y = event.offsetY;
        return this._in && ((Math.abs(this.width / 2 - Math.abs(this.x - x)) < this.boder_range) ||
               (Math.abs(this.height / 2 - Math.abs(this.y - y)) < this.boder_range));
    }

    _update_position(){
        if(this.status == 20){
            var dx = this.event.offsetX - this.mouse_down_event.offsetX;
            var dy = this.event.offsetY - this.mouse_down_event.offsetY;
            var p1 = [this.x - this.width / 2, this.y - this.height / 2];
            var p2 = [this.x + this.width / 2, this.y + this.height / 2];
            p2[0] += dx;
            p2[1] += dy;
            this.width = Math.abs(p1[0] - p2[0]);
            this.height = Math.abs(p1[1] - p2[1]);
            this.x = (p1[0] + p2[0]) / 2;
            this.y = (p1[1] + p2[1]) / 2;
        } else if (this.status == 10) {
            var dx = this.event.offsetX - this.mouse_down_event.offsetX;
            var dy = this.event.offsetY - this.mouse_down_event.offsetY;
            this.x += dx;
            this.y += dy;
        }
    }

    handleEvent(event) {
        this.event = event;
        var block_event = false;
        switch(this.status){
        case 0:
            if (event.type == 'mousemove'){}
            if (event.type == 'mouseup'){}
            if (event.type == 'mousedown'){
                if(this._in){
                    if (this._boder) {
                        this.status = 20;
                        this.mouse_down_event = event;
                        block_event = true;
                    } else {
                        this.status = 10;
                        this.mouse_down_event = event;
                        block_event = true;
                    }
                }
            }
            break;
        case 10:
            block_event = true;
            if (event.type == 'mouseup') {
                this._update_position();
                block_event = false;
                this.status = 0;
            }
            break;
        case 20:
            block_event = true;
            if (event.type == 'mouseup') {
                this._update_position();
                block_event = false;
                this.status = 0;
                console.log('cancel event');
            }
        };
        return block_event;
    }
}

var app = new Vue({
    el: "#app",
    data: {
        canvas: null,
        ctx: null,
        image: null,

        current_box:-1,
        boxes:[],
        adding_box: false,

        file_to_upload : null,

        current_image: null,

        current_page:1,
        images_per_page:9,
        num_page:0,
        images_this_page: [],

        sessions:[],
        current_session:{"session_name":'null'},

        new_session_name: null,

        show_existing_images: false,
    },
    methods: {
        remove_selected: function(){
            console.info('delete box');
            if (this.selected_box < 0) return;
            this.boxes.splice(this.selected_box, 1);
            this.selected_box = Math.min(this.selected_box, this.boxes.length-1);
        },
        update_session : function(){
            Vue.http.get(
                '/api/sessions').
            then(response => {
                console.log(response.data);
                result = response.data;
                this.sessions = response.data.sessions;
            }, response => {
                console.log(response);
            })
        },
        add_session: function(){
            console.log('adding fuck session');
            Vue.http.post('/api/session', {
                session_name:this.new_session_name,
            }).then(response => {
                console.log(response.data);
                this.update_session();
            })
        },
        _add_boxes: function(box_xywh) {
            var box = new Box();
            box.x = box_xywh[1] * this.canvas.width;
            box.width = box_xywh[3] * this.canvas.width;
            box.y = box_xywh[0] * this.canvas.height;
            box.height = box_xywh[2] * this.canvas.height;
            this.boxes.push(box);
            return box;
        },
        handleNewImage: function (record) {
            this.current_image = record;
            // set image
            url = record.url;
            var img = new Image();
            img.onload = function() {
                const width = 500;
                console.log(this);
                console.log(this.width);
                app.canvas.height = Math.floor(app.canvas.width / this.width * this.height)
                app.render();
            };
            img.src = url;
            this.image = img;
            // add boxes
            if(this.show_existing_images){
                Vue.http.get(
                    '/api/annotation/' + this.current_image.id
                ).then(response => {
                    console.log(response.data);
                    app.boxes = [];
                    for(var i=0; i < response.data.length; i += 1){
                        var box = app._add_boxes(response.data[i].points)
                        box.record = response.data[i];
                    }
                    app.render();
                })
            } else {
                Vue.http.get(
                    '/api/get_result/' + this.current_image.id
                ).then(response => {
                    console.log(response.data);
                    app.boxes = [];
                    for(var i=0; i < response.data.result.length; i += 1){
                        console.info(response.data.result[i])
                        var box = app._add_boxes(response.data.result[i].box_xywh)
                        box.detect = response.data[i];
                    }
                    app.render();
                });
            }
        },
        render: function () {
            if (this.image != null) {
                this.canvas.height = Math.floor(this.canvas.width / this.image.width * this.image.height);
                this.ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);
            }
            for(var i=0; i < this.boxes.length; i += 1){
                this.ctx.save();
                this.ctx.strokeStyle = "green";
                if (this.current_box <0) {
                    if (this.boxes[i]._boder) {
                        this.ctx.strokeStyle = 'red';
                    } else if (this.boxes[i]._in){
                        this.ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
                        this.ctx.fillRect(...this.boxes[i].top_left);
                    }
                }
                if (this.selected_box == i){
                    this.ctx.setLineDash([2, 2]);
                }
                this.ctx.strokeRect(
                    ...this.boxes[i].top_left
                );
                this.ctx.restore();
            }
        },
        handle_mouse_event (event) {
            if (this.adding_box){
            if (event.type == 'mousedown'){
                console.log(event);
                this.adding_box = false;
                newbox = new Box();
                newbox.x = event.offsetX;
                newbox.y = event.offsetY;
                newbox.mouse_down_event = event;
                newbox.status = 20;
                newbox.dx = 1;
                newbox.dy = 1;
                this.boxes.push(newbox)
            }}
            if (this.current_box < 0) {
                for(var i = 0; i < this.boxes.length; i += 1){
                    if (this.boxes[i].handleEvent(event)){
                        this.current_box = i;
                        this.selected_box = i;
                        break;
                    }
                }
            } else {
                if (!this.boxes[this.current_box].handleEvent(event)) {
                    this.current_box = -1;
                }
            }
            event.preventDefault();
            this.render();
        },
        _normal_post_data: function name(url, form_data) {
            form_data.set('session_name', this.current_session.session_name)
            Vue.http.post(url, form_data).then(response => {
                console.log(response);
            }, response => {
                console.log(response);
            });
        },
        submit: function () {
            console.log('submit!');
            var form_data = new FormData();
            for(var i = 0; i < this.file_to_upload.length; i += 1){
                form_data.append(i, this.file_to_upload[i])
                if (i != 0 && i % 10 == 0) {
                    this._normal_post_data('/api/add_img', form_data);
                    form_data = new FormData();
                }
            }
            if (i % 10 !=0){
                this._normal_post_data('/api/add_img', form_data);
            }
        },
        submit_anno: function() {
            console.log('submit annotation');
            var post_data = [];
            for (var i = 0; i < this.boxes.length; i += 1) {
                var bbox = this.boxes[i].bbox;
                bbox[0] /= this.canvas.width;
                bbox[2] /= this.canvas.width;
                bbox[1] /= this.canvas.height;
                bbox[3] /= this.canvas.height;
                bbox = [bbox[1], bbox[0], bbox[3], bbox[2]];
                var rec = null;
                if (this.boxes[i].record != null) {
                    rec = this.boxes[i].record;
                    rec.points = bbox;
                }else{
                    rec = {
                        image_id: this.current_image.id,
                        session_name: this.current_image.session_name,
                        points: bbox,
                    };
                }
                post_data.push(rec);
            }
            if (post_data.length == 0){
                post_data.push({
                    image_id: this.current_image.id,
                    session_name: this.current_image.session_name,
                    points: [-1,-1,-1,-1],
                });
            }
            console.log(post_data);
            Vue.http.post('/api/annotation', post_data).then(
                response => {
                    console.log(response.data);
                    app._update_images_this_page();
                }, response => {
                    console.log(response.data);
                }
            );
        },
        _keyboard_event: function(event){
            console.log(event);
            if(event.key == "d"){
                this.remove_selected();
            } else if(event.key == 'a'){
                this.adding_box = true;
            } else if(event.key == 's'){
                this.adding_box = false;
            } else if(event.key == 'q'){
                this.selected_box = (this.selected_box + 1) % this.boxes.length;
            } else if(event.key == 'w'){
                this.submit_anno();
            }
            this.render();
        },
        _update_images_this_page(){
            var url = '/api/image_list/'+this.current_session.session_name+'/'+this.current_page+'/'+this.images_per_page;
            if (this.show_existing_images){
                url = '/api/image_list_existing/'+this.current_session.session_name+'/'+this.current_page+'/'+this.images_per_page;
            }
            Vue.http.get(url)
            .then(response => {
                console.log(response.data);
                this.num_page = response.data.num_page;
                this.images_this_page = response.data.result;
                this.handleNewImage(this.images_this_page[0]);
                return response.data.result;
            })
        }
    },
    asyncComputed: {
        images_this_page() {
            this._update_images_this_page();
        },
    },
    computed: {

    },
    watch: {
        // image_to_annotate: function(val) {
        // }
    },
    mounted() {
        this.canvas = document.getElementById('canvas');
        this.canvas.onmousedown = this.handle_mouse_event;
        this.canvas.onmouseup = this.handle_mouse_event;
        this.canvas.onmousemove = this.handle_mouse_event;
        this.canvas.oncontextmenu=this.n;
        this.ctx = this.canvas.getContext('2d');
        document.addEventListener("keydown", function(event){
            app._keyboard_event(event);
        }, false);
        this.update_session();
        console.log('init!')
    }
});
