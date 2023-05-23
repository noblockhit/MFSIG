function check_and_adjust_flex_orientation() {
    html = $("html");
    screen_aspect_ratio = html.width() / html.height();

    if (screen_aspect_ratio > img_aspect_ratio) {
    } else {
    }

    img_container = $("#image-container");
    img_container_aspect_ratio = img_container.width() / img_container.height();

    if (img_container_aspect_ratio > img_aspect_ratio) {
        console.log(1, img_container.height(), img.height())
        new_height = img_container.height();
        new_width = new_height * img_aspect_ratio;
        img.width(new_width);
        img.height(new_height);
    } else {
        console.log(2, img_container.height(), img.height())
        new_width = img_container.width();
        new_height = new_width / img_aspect_ratio;
        img.width(new_width);
        img.height(new_height);
    }
}


html = $("html");
screen_aspect_ratio = html.width() / html.height();


$(window).on("load", () => {
    img = $("#live-image");
    img_aspect_ratio = img.width() / img.height();

    console.log(screen_aspect_ratio , img_aspect_ratio)
    if (screen_aspect_ratio > img_aspect_ratio) {
        href = "static/desktop_style.css"
    } else {
        href = "static/mobile_style.css"
    }
    var link = document.createElement('link');
    link.setAttribute("rel", "stylesheet");
    link.setAttribute("type", "text/css");
    link.onload = check_and_adjust_flex_orientation;
    link.setAttribute("href", href);
    document.getElementsByTagName("head")[0].appendChild(link);
    
});

addEventListener("resize", check_and_adjust_flex_orientation);


