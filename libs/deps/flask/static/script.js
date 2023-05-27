function check_and_adjust_flex_orientation() {
    html = $("html");
    screen_aspect_ratio = html.width() / html.height();

    img_container = $("#image-container");
    img_container_aspect_ratio = img_container.width() / img_container.height();

    if (img_container_aspect_ratio > img_aspect_ratio) {
        console.log(1, img_container.height(), img.height());
        new_height = img_container.height();
        new_width = new_height * img_aspect_ratio;
        img.width(new_width);
        img.height(new_height);
    } else {
        console.log(2, img_container.height(), img.height());
        new_width = img_container.width();
        new_height = new_width / img_aspect_ratio;
        img.width(new_width);
        img.height(new_height);
    }
}

function update_curr_pos() {
    console.log("getting curr position");
    $.ajax({
        url: "/microscope/current",
        type: "GET",
        success: function (data) {
            $("#current").html(`Current: ${data}`);
        },
        error: function (err) {
            console.log(err);
        },
    });
}

function update_start_pos() {
    $.ajax({
        url: "/microscope/start",
        type: "GET",
        success: function (data) {
            $("#start").html(`Start: ${data}`);
        },
        error: function (err) {
            console.log(err);
        },
    });
}

function update_end_pos() {
    $.ajax({
        url: "/microscope/end",
        type: "GET",
        success: function (data) {
            $("#ending").html(`Ending: ${data}`);
        },
        error: function (err) {
            console.log(err);
        },
    });
}

html = $("html");
screen_aspect_ratio = html.width() / html.height();

$(window).on("load", () => {
    img = $("#live-image");
    img_aspect_ratio = img.width() / img.height();
    check_and_adjust_flex_orientation();

    // button events

    mvActiveUp = false;
    function check_for_move_held_up() {
        if (!mvActiveUp) return;
        $.ajax({
            url: `/microscope/up`,
            success: function (data) {
                $("#current").html(`Current: ${data}`);
            },
            error: function (data) {
                console.log(data);
            },
        });
    }

    mvActiveDown = false;
    function check_for_move_held_down() {
        if (!mvActiveDown) return;
        $.ajax({
            url: `/microscope/down`,
            success: function (data) {
                $("#current").html(`Current: ${data}`);
            },
            error: function (data) {
                console.log(data);
            },
        });
    }

    // up button
    mdUpEvt = false;
    mvUpInterval = null;
    $("#move-up").on("pointerdown", function () {
        if (mdUpEvt) return;
        mdUpEvt = true;
        mvActiveUp = true;
        $.ajax({
            url: "/microscope/up",
            success: function (data) {
                $("#current").html(`Current: ${data}`);
            },
            error: function (data) {
                console.log(data);
            },
        });
        clearInterval(mvUpInterval);
        setTimeout(() => {
            mvUpInterval = setInterval(check_for_move_held_up, 100);
            mdUpEvt = false;
        }, 500);
    });

    $("#move-up").on("mouseup mouseleave pointerup", function () {
        mvActiveUp = false;
        clearInterval(mvUpInterval);
    });

    // down button

    mdDownEvt = false;
    mvDownInterval = null;
    $("#move-down").on("pointerdown", function () {
        if (mdDownEvt) return;
        mdDownEvt = true;
        mvActiveDown = true;
        $.ajax({
            url: "/microscope/down",
            success: function (data) {
                $("#current").html(`Current: ${data}`);
            },
            error: function (data) {
                console.log(data);
            },
        });
        clearInterval(mvDownInterval);
        setTimeout(() => {
            mvDownInterval = setInterval(check_for_move_held_down, 100);
            mdDownEvt = false;
        }, 500);
    });

    $("#move-down").on("mouseup mouseleave pointerup", function () {
        mvActiveDown = false;
        clearInterval(mvDownInterval);
    });

    // manual button updates
    $("#current").on("pointerdown", update_curr_pos);
    $("#start").on("pointerdown", update_start_pos);
    $("#ending").on("pointerdown", update_end_pos);

    $("#set-start").on("pointerdown", () => {
        $.post("/microscope/start", (async = false));
        update_start_pos();
    });

    $("#set-ending").on("pointerdown", () => {
        $.post("/microscope/end", (async = false));
        update_end_pos();
    });

    $("#move-start").on("pointerdown", () => {
        $.get("/microscope/move/start", (async = false));
        update_curr_pos();
    });

    $("#move-ending").on("pointerdown", () => {
        $.get("/microscope/move/end", (async = false));
        update_curr_pos();
    });
});

addEventListener("resize", check_and_adjust_flex_orientation);
